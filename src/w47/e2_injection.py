"""E2 - controlled scenarios on the control period (SPEC section 6, E2).

Every run draws an onset day ``o`` and builds a 42-day window [o - 14, o + 28) of daily chunks.
The frozen forecaster's real predictions are used; only the scenario changes:

* ``control``          nothing changes (false-alarm baseline)
* ``cs_harm``/``cs_benign``  after o, each day = 24 real (x, y, y_pred) rows resampled from the
  control pool with weights exp(beta * z(x)), z = +/- standardised raw ``lag_168``. The weight
  depends on x only, so P(y | x) is unchanged by construction: pure covariate shift.
* ``cd_bias``          y' = (1 + d) y after o, x and y_pred frozen: pure concept drift (mean)
* ``cd_noise``         y' = y exp(s eps - s^2 / 2), x and y_pred frozen: pure concept drift (noise)
* ``prop``             the whole series is multiplied by (1 + d) from o, features and predictions
  are recomputed: the realistic autoregressive case, P(y | x) first, P(x) after 168 h.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from w47.model import add_predictions, build_frame
from w47.monitoring import Monitor, input_drift_alerts
from w47.pipeline import RESULTS, forecaster, forecasts, load_config, series, slice_period


@dataclass(frozen=True)
class Scenario:
    family: str
    param: float | None = None

    @property
    def name(self) -> str:
        return self.family if self.param is None else f"{self.family}({self.param:+g})"


def scenarios(cfg: dict) -> list[Scenario]:
    inj = cfg["injection"]
    out = [Scenario("control")]
    out += [Scenario("cs_harm", b) for b in inj["covariate_shift_beta"]]
    out += [Scenario("cs_benign", b) for b in inj["covariate_shift_beta"]]
    out += [Scenario("cd_bias", d) for d in inj["concept_bias"]]
    out += [Scenario("cd_noise", s) for s in inj["concept_noise"]]
    out += [Scenario("prop", d) for d in inj["propagated_bias"]]
    return out


def onsets(cfg: dict, days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    inj = cfg["injection"]
    rng = np.random.default_rng(inj["seed"])
    idx = rng.integers(inj["pre_days"], len(days) - inj["post_days"] + 1, size=inj["n_runs"])
    return [days[i] for i in idx]


def build_window(sc: Scenario, onset: pd.Timestamp, pool: pd.DataFrame, cfg: dict,
                 rng: np.random.Generator) -> pd.DataFrame:
    inj = cfg["injection"]
    start = onset - pd.Timedelta(days=inj["pre_days"])
    end = onset + pd.Timedelta(days=inj["post_days"])
    win = slice_period(pool, start, end).copy()
    post = win.index >= onset

    if sc.family == "control":
        return win
    if sc.family in ("cs_harm", "cs_benign"):
        z = (pool["lag_168"] - pool["lag_168"].mean()) / pool["lag_168"].std()
        sign = 1.0 if sc.family == "cs_harm" else -1.0
        w = np.exp(sign * sc.param * z.to_numpy())
        w /= w.sum()
        pre_part = win[~post]
        post_days = pd.date_range(onset, end - pd.Timedelta(hours=1), freq="h")
        draw = rng.choice(len(pool), size=len(post_days), replace=True, p=w)
        res = pool.iloc[draw].copy()
        res.index = post_days
        res["timestamp"] = post_days  # only used for chunking; x keeps its own calendar values
        return pd.concat([pre_part, res])
    if sc.family == "cd_bias":
        win.loc[post, "y_true"] = win.loc[post, "y_true"] * (1 + sc.param)
        return win
    if sc.family == "cd_noise":
        eps = rng.standard_normal(post.sum())
        win.loc[post, "y_true"] = win.loc[post, "y_true"] * np.exp(sc.param * eps
                                                                    - sc.param**2 / 2)
        return win
    if sc.family == "prop":
        y = series().copy()
        y[y.index >= onset] *= 1 + sc.param
        hist = y[(y.index >= start - pd.Timedelta(days=60)) & (y.index < end)]
        frame = add_predictions(build_frame(hist), forecaster())
        return slice_period(frame, start, end)
    raise ValueError(sc.family)


def run_shard(shard: int = 0, n_shards: int = 1, n_runs: int | None = None) -> Path:
    """Run every ``n_shards``-th job starting at ``shard`` in this process.

    Shards are plain OS processes launched by the Makefile (``make e2``), not a multiprocessing
    pool: LightGBM inside spawned pool workers spun for >25 min on a 1 s fit (see
    IMPLEMENTATION_NOTES, P5). Each shard writes its own parquet; ``merge_shards`` joins them.
    """
    cfg = load_config()
    if n_runs:
        cfg["injection"]["n_runs"] = n_runs
    df = forecasts()
    pool = slice_period(df, *cfg["periods"]["control"])
    t0 = time.time()
    mon = Monitor(with_replica=False).fit(slice_period(df, *cfg["periods"]["reference"]))
    print(f"[shard {shard}] monitor fitted in {time.time() - t0:.0f}s", flush=True)
    days = pd.date_range(pool.index.min().normalize(), pool.index.max().normalize(), freq="D")
    jobs = [(sc, i, o) for sc in scenarios(cfg) for i, o in enumerate(onsets(cfg, days))]
    mine = jobs[shard::n_shards]
    tabs = []
    for j, (sc, run_id, onset) in enumerate(mine):
        # crc32, not hash(): str hashes are salted per process
        rng = np.random.default_rng(zlib.crc32(f"{sc.name}|{run_id}".encode()))
        tab = mon.run(build_window(sc, onset, pool, cfg, rng))
        tab["n_input_drift"] = input_drift_alerts(tab)
        tab["scenario"], tab["family"], tab["param"] = sc.name, sc.family, sc.param
        tab["run"], tab["onset"] = run_id, onset
        tab["rel_day"] = (tab.index - onset).days
        tabs.append(tab)
        if (j + 1) % 10 == 0:
            print(f"[shard {shard}] {j + 1}/{len(mine)} jobs, {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(tabs)
    out.index.name = "day"
    keep = [c for c in out.columns if not c.startswith("uni_") or c.startswith("uni_alert_")
            or c.endswith("_wasserstein")]
    path = RESULTS / "cache" / f"e2_runs_shard{shard}of{n_shards}.parquet"
    out[keep].to_parquet(path)
    print(f"[shard {shard}] done: {len(mine)} jobs in {time.time() - t0:.0f}s", flush=True)
    return path


def merge_shards(n_shards: int) -> pd.DataFrame:
    parts = [pd.read_parquet(RESULTS / "cache" / f"e2_runs_shard{i}of{n_shards}.parquet")
             for i in range(n_shards)]
    out = pd.concat(parts)
    out.to_parquet(RESULTS / "cache" / "e2_runs.parquet")
    print(f"E2 merged: {out.groupby(['scenario', 'run']).ngroups} runs")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="E2 injected scenarios")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-runs", type=int, default=None, help="override protocol (smoke tests)")
    ap.add_argument("--merge", action="store_true", help="merge shard files and exit")
    a = ap.parse_args()
    if a.merge:
        merge_shards(a.n_shards)
    else:
        run_shard(a.shard, a.n_shards, a.n_runs)
