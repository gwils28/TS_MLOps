"""Invariants of the injected scenarios (E2): what must change and what must not."""

import numpy as np
import pandas as pd
import pytest

from w47.e2_injection import Scenario, build_window
from w47.e3_detection import _persist
from w47.features import FEATURES

CFG = {"injection": {"pre_days": 3, "post_days": 4}}


@pytest.fixture
def pool():
    idx = pd.date_range("2019-07-01", periods=24 * 20, freq="h")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f: rng.normal(size=len(idx)) for f in FEATURES}, index=idx)
    df["lag_168"] = rng.uniform(500, 20000, len(idx))
    df["y_true"] = df["lag_168"] * rng.uniform(0.9, 1.1, len(idx))
    df["y_pred"] = df["lag_168"]
    df["timestamp"] = idx
    return df


ONSET = pd.Timestamp("2019-07-08")


def test_concept_drift_freezes_inputs_and_predictions(pool):
    base = build_window(Scenario("control"), ONSET, pool, CFG, np.random.default_rng(0))
    for sc in (Scenario("cd_bias", -0.2), Scenario("cd_noise", 0.3)):
        win = build_window(sc, ONSET, pool, CFG, np.random.default_rng(0))
        pd.testing.assert_frame_equal(win[FEATURES + ["y_pred"]], base[FEATURES + ["y_pred"]])
        pre = win.index < ONSET
        assert np.allclose(win.loc[pre, "y_true"], base.loc[pre, "y_true"])
        assert not np.allclose(win.loc[~pre, "y_true"], base.loc[~pre, "y_true"])


def test_bias_is_exact(pool):
    win = build_window(Scenario("cd_bias", -0.3), ONSET, pool, CFG, np.random.default_rng(0))
    base = build_window(Scenario("control"), ONSET, pool, CFG, np.random.default_rng(0))
    post = win.index >= ONSET
    assert np.allclose(win.loc[post, "y_true"], 0.7 * base.loc[post, "y_true"])


def test_covariate_shift_keeps_real_xy_pairs(pool):
    """Resampled rows are real (x, y, y_pred) triples of the pool: P(y|x) cannot change."""
    win = build_window(Scenario("cs_harm", 2.0), ONSET, pool, CFG, np.random.default_rng(0))
    post = win[win.index >= ONSET]
    key = ["lag_168", "y_true", "y_pred"]
    pool_keys = set(map(tuple, pool[key].round(6).to_numpy()))
    assert all(tuple(r) in pool_keys for r in post[key].round(6).to_numpy())
    assert post["lag_168"].mean() > pool["lag_168"].mean()  # tilted towards high demand


def test_benign_shift_goes_the_other_way(pool):
    win = build_window(Scenario("cs_benign", 2.0), ONSET, pool, CFG, np.random.default_rng(0))
    assert win[win.index >= ONSET]["lag_168"].mean() < pool["lag_168"].mean()


def test_window_length(pool):
    win = build_window(Scenario("cs_harm", 1.0), ONSET, pool, CFG, np.random.default_rng(0))
    assert len(win) == 24 * 7 and win.index.is_monotonic_increasing


def test_persistence():
    f = pd.Series([False, True, False, True, True, True, False])
    assert _persist(f, 2).tolist() == [False, False, False, False, True, True, False]
