"""Feature engineering under the issue-time constraint.

A forecast for target hour ``t`` is issued at ``t - H`` with ``H = 168``. Every feature of row
``t`` therefore uses only values observed at or before ``t - 168``. ``tests/test_features.py``
corrupts the future of the series and checks that no feature moves.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

H = 168

LAG_FEATURES = ["lag_168", "lag_192", "lag_336", "lag_504", "lag_672"]
ROLL_FEATURES = ["mean_same_hour_4w", "roll24_at168", "roll168_at168", "trend_ratio"]
CALENDAR_FEATURES = ["hour", "dayofweek", "month", "is_holiday"]

CONTINUOUS_FEATURES = LAG_FEATURES + ROLL_FEATURES
CATEGORICAL_FEATURES = CALENDAR_FEATURES
FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES


def us_holidays(index: pd.DatetimeIndex) -> np.ndarray:
    years = range(index.year.min(), index.year.max() + 1)
    hol = pd.DatetimeIndex(list(holidays.country_holidays("US", years=years).keys()))
    return index.normalize().isin(hol).astype(int)


def make_features(y: pd.Series, min_lag: int = H) -> pd.DataFrame:
    """Build the feature frame for every hour of ``y`` (hourly, complete grid).

    ``min_lag`` exists so that the leakage test can assert the constraint; the protocol uses 168.
    Rows without enough history contain NaN and are dropped by callers.
    """
    if min_lag < H:
        raise ValueError("protocol requires every lag >= 168 h")
    y = y.astype(float)
    X = pd.DataFrame(index=y.index)
    for lag in (168, 192, 336, 504, 672):
        X[f"lag_{lag}"] = y.shift(lag)
    X["mean_same_hour_4w"] = X[["lag_168", "lag_336", "lag_504", "lag_672"]].mean(axis=1)
    shifted = y.shift(H)  # value at t - 168 is the most recent observable one
    X["roll24_at168"] = shifted.rolling(24).mean()
    X["roll168_at168"] = shifted.rolling(168).mean()
    X["trend_ratio"] = X["roll168_at168"] / y.shift(2 * H).rolling(168).mean()
    X["hour"] = y.index.hour
    X["dayofweek"] = y.index.dayofweek
    X["month"] = y.index.month
    X["is_holiday"] = us_holidays(y.index)
    return X


def period_mask(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    """Half-open [start, end) mask."""
    return (index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))
