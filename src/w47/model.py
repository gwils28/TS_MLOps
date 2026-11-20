"""The monitored forecaster (LightGBM on log1p target) and its baselines."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from w47.features import CATEGORICAL_FEATURES, FEATURES, LEVEL, make_features, period_mask


@dataclass
class Forecaster:
    """LightGBM on a level-relative target.

    target = log1p(y_t) - log1p(level), level = mean of the 168 h ending at t - 168 (observable
    at issue time). Prediction = expm1(model + log1p(level)). ``target_transform="log1p"`` keeps
    the original v0 behaviour (absolute log target) for the record of deviation D1.
    """

    params: dict
    target_transform: str = "log_ratio_to_level"
    model: lgb.LGBMRegressor | None = None

    def _anchor(self, X: pd.DataFrame) -> np.ndarray:
        if self.target_transform == "log_ratio_to_level":
            return np.log1p(X[LEVEL].to_numpy())
        return np.zeros(len(X))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster:
        target = np.log1p(y.to_numpy()) - self._anchor(X)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X[FEATURES], target, categorical_feature=CATEGORICAL_FEATURES)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict(X[FEATURES]) + self._anchor(X)
        return np.clip(np.expm1(raw), 0, None)


def build_frame(y: pd.Series) -> pd.DataFrame:
    """Features + target on the complete hourly grid (rows lacking history are dropped)."""
    X = make_features(y)
    df = X.copy()
    df["y_true"] = y
    df["timestamp"] = y.index
    return df.dropna(subset=FEATURES + [LEVEL])


def train_forecaster(df: pd.DataFrame, cfg: dict) -> Forecaster:
    tr = df[period_mask(df.index, *cfg["periods"]["train"])]
    return Forecaster(cfg["model"]["lgbm_params"], cfg["model"]["target_transform"]).fit(
        tr, tr["y_true"]
    )


def add_predictions(df: pd.DataFrame, fc: Forecaster) -> pd.DataFrame:
    out = df.copy()
    out["y_pred"] = fc.predict(out)
    out["y_naive"] = out["lag_168"]  # seasonal weekly naive, the credibility baseline
    out["y_naive4w"] = out["mean_same_hour_4w"]
    return out


def point_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / np.maximum(y_true, 1.0))),
        "bias": float(np.mean(-err)),  # >0: over-forecast
        "n": int(len(err)),
    }
