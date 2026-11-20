"""E1 - real shock: monitor the frozen forecaster from 2019-07 to 2021-12."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from w47.model import point_metrics
from w47.monitoring import Monitor, input_drift_alerts
from w47.pipeline import TABLES, forecasts, load_config, slice_period

PHASES = [  # SPEC section 3, fixed by the calendar
    ("P0 pré-choc", "2019-07-01", "2020-03-09"),
    ("P1 transition", "2020-03-09", "2020-03-30"),
    ("P2 confinement", "2020-03-30", "2020-06-08"),
    ("P3 reprise", "2020-06-08", "2021-07-01"),
    ("P4 nouveau régime", "2021-07-01", "2022-01-01"),
]


def phase_of(days: pd.DatetimeIndex) -> pd.Series:
    lab = pd.Series(index=days, dtype=object)
    for name, a, b in PHASES:
        lab[(days >= a) & (days < b)] = name
    return lab


def run(chunk_period: str = "D") -> pd.DataFrame:
    cfg = load_config()
    df = forecasts()
    ref = slice_period(df, *cfg["periods"]["reference"])
    ana = slice_period(df, cfg["periods"]["control"][0], cfg["periods"]["covid"][1])
    t0 = time.time()
    mon = Monitor(chunk_period=chunk_period, metrics=("mae", "rmse", "mape")).fit(ref)  # mape: D2
    table = mon.run(ana)
    table["n_input_drift"] = input_drift_alerts(table)
    table["phase"] = phase_of(table.index)
    suffix = "" if chunk_period == "D" else f"_{chunk_period}"
    table.to_csv(TABLES / f"e1_daily{suffix}.csv")
    # reference-period diagnostics (in-sample NannyML vs replica vs cross-fitted)
    refdiag = pd.DataFrame({"real_mae": mon.real_ref["mae"], "rep_mae": mon.rep_ref["mae"],
                            "cf_mae": mon.cf_ref["mae"], "real_rmse": mon.real_ref["rmse"],
                            "rep_rmse": mon.rep_ref["rmse"], "cf_rmse": mon.cf_ref["rmse"]})
    refdiag.to_csv(TABLES / f"e1_reference_chunks{suffix}.csv")
    print(f"E1 monitors fitted and run in {time.time() - t0:.0f}s")
    return table


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Per-phase agreement between estimated and realized MAE."""
    rows = []
    for name, *_ in PHASES:
        t = table[table.phase == name]
        if t.empty:
            continue
        rows.append({
            "phase": name,
            "jours": len(t),
            "MAE réalisée (moy.)": t.real_mae.mean(),
            "MAE estimée DLE (moy.)": t.est_mae.mean(),
            "ratio estimée/réalisée": t.est_mae.mean() / t.real_mae.mean(),
            "corr. de rang (Spearman)": t[["est_mae", "real_mae"]].corr("spearman").iloc[0, 1],
            "réalisée dans la bande ±3 se": float(np.mean(
                (t.real_mae >= t.est_mae - 3 * t.est_mae_se)
                & (t.real_mae <= t.est_mae + 3 * t.est_mae_se))),
            "alertes réalisées": int(t.real_mae_alert.sum()),
            "alertes DLE": int(t.est_mae_alert.sum()),
            "alertes PCA": int(t.pca_alert.sum()),
            "jours ≥2 dérives d'entrée": int((t.n_input_drift >= 2).sum()),
            "hors support (moy.)": t.support_out.mean(),
        })
    return pd.DataFrame(rows)


def period_metrics() -> pd.DataFrame:
    cfg = load_config()
    df = forecasts()
    rows = []
    for k, (a, b) in cfg["periods"].items():
        d = slice_period(df, a, b)
        for col, name in [("y_pred", "LightGBM (surveillé)"), ("y_naive", "naïve S7 (lag 168)"),
                          ("y_naive4w", "moyenne 4 semaines")]:
            rows.append({"période": k, "modèle": name, **point_metrics(d.y_true, d[col])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = run()
    s = summarise(t)
    s.to_csv(TABLES / "e1_phases.csv", index=False)
    period_metrics().to_csv(TABLES / "forecaster_metrics.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())
