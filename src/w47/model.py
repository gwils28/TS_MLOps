"""The monitored forecaster (LightGBM on log1p target) and its baselines."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from w47.features import CATEGORICAL_FEATURES, FEATURES, make_features, period_mask


@dataclass
class Forecaster:
    params: dict
    target_transform: str = "log1p"
    model: lgb.LGBMRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster:
        target = np.log1p(y) if self.target_transform == "log1p" else y
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X[FEATURES], target, categorical_feature=CATEGORICAL_FEATURES)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict(X[FEATURES])
        pred = np.expm1(raw) if self.target_transform == "log1p" else raw
        return np.clip(pred, 0, None)


def build_frame(y: pd.Series) -> pd.DataFrame:
    """Features + target on the complete hourly grid (rows lacking history are dropped)."""
    X = make_features(y)
    df = X.copy()
    df["y_true"] = y
    df["timestamp"] = y.index
    return df.dropna(subset=FEATURES)


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
