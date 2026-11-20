"""E5 (exploratory, SPEC section 6) - retraining too early after the shock.

At R = S + k weeks (S = 2020-03-09, k in {1, 2, 4, 8}), with labels known up to R (L = 0), compare
on the targets [R, R + 8 weeks):

* ``frozen``      the production model, untouched;
* ``rolling_1y``  refit on the 52 weeks of labelled rows ending at R (old world dominates);
* ``post_only``   refit on the k post-shock weeks only (new world, very few rows);
* ``weighted``    refit on the 52 weeks with post-shock rows weighted x10.

The forecaster is otherwise identical (features, hyperparameters, level-relative target).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from w47.model import Forecaster, point_metrics
from w47.pipeline import TABLES, forecasts, load_config

SHOCK = pd.Timestamp("2020-03-09")
WEEKS = (1, 2, 4, 8)
EVAL_WEEKS = 8


def _fit(cfg, rows: pd.DataFrame, weight: np.ndarray | None = None) -> Forecaster:
    fc = Forecaster(cfg["model"]["lgbm_params"], cfg["model"]["target_transform"])
    if weight is None:
        return fc.fit(rows, rows["y_true"])
    import lightgbm as lgb

    from w47.features import CATEGORICAL_FEATURES, FEATURES

    target = np.log1p(rows["y_true"].to_numpy()) - fc._anchor(rows)
    fc.model = lgb.LGBMRegressor(**cfg["model"]["lgbm_params"])
    fc.model.fit(rows[FEATURES], target, sample_weight=weight,
                 categorical_feature=CATEGORICAL_FEATURES)
    return fc


def run() -> pd.DataFrame:
    cfg = load_config()
    df = forecasts()
    out = []
    for k in WEEKS:
        R = SHOCK + pd.Timedelta(weeks=k)
        test = df[(df.index >= R) & (df.index < R + pd.Timedelta(weeks=EVAL_WEEKS))]
        hist = df[(df.index >= R - pd.Timedelta(weeks=52)) & (df.index < R)]
        post = hist[hist.index >= SHOCK]
        variants = {
            "frozen (aucun réentraînement)": test["y_pred"].to_numpy(),
            "rolling 52 semaines": _fit(cfg, hist).predict(test),
            "post-choc seul": _fit(cfg, post).predict(test),
            "52 semaines, post-choc ×10": _fit(
                cfg, hist, np.where(hist.index >= SHOCK, 10.0, 1.0)).predict(test),
        }
        for name, pred in variants.items():
            m = point_metrics(test["y_true"], pred)
            wape = float(np.abs(test["y_true"] - pred).sum() / test["y_true"].sum())
            out.append({"k (semaines après le choc)": k, "réentraîné le": R.date(),
                        "lignes post-choc": len(post), "variante": name, "MAE": m["mae"],
                        "WAPE": wape, "biais": m["bias"]})
    res = pd.DataFrame(out)
    res.to_csv(TABLES / "e5_retrain.csv", index=False)
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(run().round(3).to_string(index=False))
