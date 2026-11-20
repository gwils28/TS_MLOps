"""E3 - recall x delay x false alarms, DriftGuard-style, on the E2 runs (SPEC section 6, E3).

Timing model (SPEC section 2), daily chunk = target day d:

* label-free signals (DLE, input drift, prediction drift) for day d are available once the last
  forecast of day d is issued, i.e. at the end of day d - 7  ->  clock time d - 6 (days);
* realized performance for day d is available at the end of day d + L  ->  clock d + 1 + L.

Delays are reported on both axes: *target* delay = first alerted target day - onset, and
*clock* delay = availability time of that alert - onset (negative = before the change happens,
possible only when the change is visible in inputs known a week ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from w47.pipeline import RESULTS, TABLES, load_config

H_DAYS = 7


def _persist(flag: pd.Series, k: int) -> pd.Series:
    """True on the k-th consecutive True (and after, while it lasts)."""
    if k <= 1:
        return flag
    return flag.astype(int).rolling(k, min_periods=k).sum().eq(k)


def reference_thresholds() -> dict[str, float]:
    """Alert thresholds derived from the reference period's daily chunks (E1 output).

    * ``d3`` / ``d2``: NannyML's rule (mean + 3 or 2 sigma of the *realized* daily MAE);
    * ``w3`` / ``w2``: the same rule on 7-day rolling means (SPEC E4, weekly sensitivity);
    * ``rel``: exploratory D3, an SLO-like relative threshold: 7-day mean above 1.25 x the
      reference mean (out-of-fold DLE estimate for DLE, realized MAE for realized).
    """
    r = pd.read_csv(TABLES / "e1_reference_chunks.csv", index_col=0, parse_dates=True)
    real, real7 = r["real_mae"], r["real_mae"].rolling(7).mean()
    return {
        "d3": real.mean() + 3 * real.std(), "d2": real.mean() + 2 * real.std(),
        "w3": real7.mean() + 3 * real7.std(), "w2": real7.mean() + 2 * real7.std(),
        "rel_real": 1.25 * real.mean(), "rel_est": 1.25 * r["cf_mae"].mean(),
    }


REALIZED_PREFIX = "réalisé"


def detector_flags(t: pd.DataFrame, k: int = 1, thr: dict | None = None) -> dict[str, pd.Series]:
    """Boolean alert series per detector for one run table (index = target day, sorted).

    DLE alerts count the upper side only: an estimate *below* the band is not a page.
    """
    thr = thr or reference_thresholds()
    est7, real7 = t["est_mae"].rolling(7).mean(), t["real_mae"].rolling(7).mean()
    uni1 = t["n_input_drift"] >= 1
    uni2 = t["n_input_drift"] >= 2
    pca = t["pca_alert"].fillna(False).astype(bool)
    pred = t.get("uni_alert_y_pred_wasserstein", pd.Series(False, index=t.index)).fillna(False)
    drift = pca | uni2
    dle = {
        "DLE seul": t["est_mae"] > t["mae_upper"],
        "DLE ±2σ": t["est_mae"] > thr["d2"],
        "DLE 7 j ±3σ": est7 > thr["w3"],
        "DLE 7 j ±2σ": est7 > thr["w2"],
        "DLE 7 j +25 % (D3)": est7 > thr["rel_est"],
    }
    flags = {
        "dérive univariée (≥1 entrée)": uni1,
        "dérive univariée (≥2 entrées)": uni2,
        "dérive multivariée (PCA)": pca,
        "dérive des prédictions": pred.astype(bool),
        **dle,
        "règle jointe (dérive ET DLE)": drift & dle["DLE seul"],
        "règle jointe 7 j +25 % (D3)": drift & dle["DLE 7 j +25 % (D3)"],
        "réalisé (labels)": t["real_mae"] > t["mae_upper"],
        "réalisé ±2σ": t["real_mae"] > thr["d2"],
        "réalisé 7 j ±3σ": real7 > thr["w3"],
        "réalisé 7 j +25 % (D3)": real7 > thr["rel_real"],
    }
    return {name: _persist(f.fillna(False).astype(bool), k) for name, f in flags.items()}


def evaluate(runs: pd.DataFrame, k: int = 1, latency_days: int = 0) -> pd.DataFrame:
    cfg = load_config()
    post_days = cfg["injection"]["post_days"]
    thr = reference_thresholds()
    rows = []
    for (scenario, run_id), t in runs.groupby(["scenario", "run"], sort=False):
        t = t.sort_index()
        onset = t["onset"].iloc[0]
        post = t["rel_day"] >= 0
        # pre-registered harm: post-onset mean realized MAE above NannyML's upper threshold
        harmful = bool(t.loc[post, "real_mae"].mean() > t["mae_upper"].iloc[0])
        # exploratory (D3): post-onset mean realized MAE at least 25 % above the pre-onset mean
        harmful_rel = bool(t.loc[post, "real_mae"].mean() >= 1.25 * t.loc[~post, "real_mae"].mean())
        for det, f in detector_flags(t, k, thr).items():
            hits = t.index[post & f.to_numpy()]
            pre_alerts = int(f[~post].sum())
            first = hits.min() if len(hits) else pd.NaT
            target_delay = (first - onset).days if len(hits) else np.nan
            if det.startswith(REALIZED_PREFIX):
                clock = target_delay + 1 + latency_days if len(hits) else np.nan
            else:
                clock = target_delay - (H_DAYS - 1) if len(hits) else np.nan
            rows.append({"scenario": scenario, "family": t["family"].iloc[0], "run": run_id,
                         "detector": det, "detected": len(hits) > 0, "harmful": harmful,
                         "harmful_rel": harmful_rel,
                         "target_delay": target_delay, "clock_delay": clock,
                         "post_alert_days": int(f[post].sum()), "pre_alert_days": pre_alerts,
                         "pre_days": int((~post).sum()), "post_days": post_days})
    return pd.DataFrame(rows)


def summary(ev: pd.DataFrame) -> pd.DataFrame:
    """Recall, delays and false-alarm rate per scenario x detector.

    False alarms: alerts on pre-onset days of every run, plus on post-onset days of control runs
    (where nothing happens), expressed per 30 days.
    """
    out = []
    ctrl = ev[ev.family == "control"]
    for (scenario, det), g in ev.groupby(["scenario", "detector"], sort=False):
        c = ctrl[ctrl.detector == det]
        fa_days = ev.loc[ev.detector == det, "pre_alert_days"].sum() + c["post_alert_days"].sum()
        exposure = ev.loc[ev.detector == det, "pre_days"].sum() + c["post_days"].sum()
        det_ = g[g.detected]
        out.append({
            "scénario": scenario,
            "détecteur": det,
            "rappel": g["detected"].mean(),
            "délai cible médian (j)": det_["target_delay"].median(),
            "délai horloge médian (j)": det_["clock_delay"].median(),
            "part de runs nocifs": g["harmful"].mean(),
            "rappel du dommage": g.loc[g.harmful, "detected"].mean() if g.harmful.any()
            else np.nan,
            "alertes sur runs non nocifs": g.loc[~g.harmful, "detected"].mean()
            if (~g.harmful).any() else np.nan,
            "part de runs nocifs (+25 %, D3)": g["harmful_rel"].mean(),
            "rappel du dommage (+25 %, D3)": g.loc[g.harmful_rel, "detected"].mean()
            if g.harmful_rel.any() else np.nan,
            "alertes sur runs non nocifs (+25 %, D3)": g.loc[~g.harmful_rel, "detected"].mean()
            if (~g.harmful_rel).any() else np.nan,
            "fausses alertes / 30 j": 30 * fa_days / exposure,
        })
    return pd.DataFrame(out)


def tracking(runs: pd.DataFrame) -> pd.DataFrame:
    """Does the DLE estimate move with the realized error? Post/pre ratios per scenario."""
    rows = []
    for scenario, g in runs.groupby("scenario", sort=False):
        pre, post = g[g.rel_day < 0], g[g.rel_day >= 7]  # skip the first week (PROP transition)
        rows.append({
            "scénario": scenario,
            "MAE réalisée avant": pre.real_mae.mean(),
            "MAE réalisée après": post.real_mae.mean(),
            "MAE estimée avant": pre.est_mae.mean(),
            "MAE estimée après": post.est_mae.mean(),
            "Δ réalisé (%)": 100 * (post.real_mae.mean() / pre.real_mae.mean() - 1),
            "Δ estimé (%)": 100 * (post.est_mae.mean() / pre.est_mae.mean() - 1),
            "part du Δ capturée": (post.est_mae.mean() - pre.est_mae.mean())
            / (post.real_mae.mean() - pre.real_mae.mean()),
            "hors support après": post.support_out.mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    runs = pd.read_parquet(RESULTS / "cache" / "e2_runs.parquet")
    tracking(runs).to_csv(TABLES / "e2_tracking.csv", index=False)
    for k in (1, 2):
        for lat in (0, 7):
            ev = evaluate(runs, k=k, latency_days=lat)
            summary(ev).to_csv(TABLES / f"e3_detection_k{k}_L{lat}.csv", index=False)
            if k == 1 and lat == 0:
                ev.to_csv(TABLES / "e3_events_k1_L0.csv", index=False)
    pd.set_option("display.width", 250)
    print(tracking(runs).round(2).to_string())
    print(pd.read_csv(TABLES / "e3_detection_k2_L0.csv").round(2).to_string())


if __name__ == "__main__":
    main()
