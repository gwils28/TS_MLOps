"""Leakage tests: features of target t must not depend on anything after t - 168 h."""

import numpy as np
import pandas as pd
import pytest

from w47.features import FEATURES, H, make_features


@pytest.fixture
def series():
    idx = pd.date_range("2019-01-01", periods=24 * 7 * 8, freq="h")
    rng = np.random.default_rng(0)
    base = 1000 + 300 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 30, len(idx))
    return pd.Series(base, index=idx)


def test_future_corruption_does_not_move_features(series):
    t = series.index[-1]  # last target hour
    X_clean = make_features(series).loc[t, FEATURES]
    corrupted = series.copy()
    corrupted[corrupted.index > t - pd.Timedelta(hours=H)] = 1e9  # everything after issue time
    X_corr = make_features(corrupted).loc[t, FEATURES]
    pd.testing.assert_series_equal(X_clean, X_corr)


def test_value_at_issue_time_is_used(series):
    """Sanity: the most recent observable value (t - 168) does move lag_168."""
    t = series.index[-1]
    s2 = series.copy()
    s2[t - pd.Timedelta(hours=H)] += 500
    assert make_features(s2).loc[t, "lag_168"] == series[t - pd.Timedelta(hours=H)] + 500


def test_min_lag_guard(series):
    with pytest.raises(ValueError):
        make_features(series, min_lag=24)
