"""Credibility guard (SPEC section 4): the forecaster must beat the seasonal naive on the
reference period. Records v0 (absolute log1p target, raw lags) against v1 (deviation D1:
level-relative target and features) so the decision is reproducible."""

from __future__ import annotations

import numpy as np
import pandas as pd

from w47.features import (
    CATEGORICAL_FEATURES,
    LEVEL,
    RAW_LAGS,
    make_features,
    period_mask,
)
from w47.model import Forecaster, point_metrics
from w47.pipeline import TABLES, load_config, series

V0_FEATURES = ["lag_168", "lag_192", "lag_336", "lag_504", "lag_672", "mean_same_hour_4w",
               "roll24_at168", "roll168_at168", "trend_ratio"] + CATEGORICAL_FEATURES


def main() -> pd.DataFrame:
    import lightgbm as lgb

    cfg = load_config()
    y = series()
    X = make_features(y).dropna()
    X["y_true"] = y
    tr = X[period_mask(X.index, *cfg["periods"]["train"])]
    ref = X[period_mask(X.index, *cfg["periods"]["reference"])]
    rows = []
    # v0: absolute log1p target on raw lags (as first pre-registered)
    m0 = lgb.LGBMRegressor(**cfg["model"]["lgbm_params"]).fit(
        tr[V0_FEATURES], np.log1p(tr["y_true"]), categorical_feature=CATEGORICAL_FEATURES)
    p0 = np.clip(np.expm1(m0.predict(ref[V0_FEATURES])), 0, None)
    rows.append({"modèle": "v0 LightGBM (cible log1p absolue)", **point_metrics(ref.y_true, p0)})
    # v1: deviation D1
    fc = Forecaster(cfg["model"]["lgbm_params"], "log_ratio_to_level").fit(tr, tr["y_true"])
    rows.append({"modèle": "v1 LightGBM (D1 : cible relative au niveau)",
                 **point_metrics(ref.y_true, fc.predict(ref))})
    rows.append({"modèle": "naïve saisonnière S7 (lag 168)",
                 **point_metrics(ref.y_true, ref["lag_168"])})
    rows.append({"modèle": "moyenne même heure 4 semaines",
                 **point_metrics(ref.y_true, ref["mean_same_hour_4w"])})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "guard_reference.csv", index=False)
    print(out.round(3).to_string())
    _ = (LEVEL, RAW_LAGS)
    return out


if __name__ == "__main__":
    main()
