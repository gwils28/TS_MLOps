"""The DLE replica must reproduce NannyML, and NannyML must see calendar columns as categorical."""

import numpy as np
import pandas as pd
import pytest

from w47.features import CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, FEATURES
from w47.monitoring import DLEReplica, _typed, input_drift_alerts


def _frame(start, days, seed, shift=0.0):
    idx = pd.date_range(start, periods=24 * days, freq="h")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(1 + shift, 0.2, len(idx)) for c in CONTINUOUS_FEATURES},
                      index=idx)
    df["hour"], df["dayofweek"] = idx.hour, idx.dayofweek
    df["month"], df["is_holiday"] = idx.month, 0
    df["y_pred"] = 1000 + 300 * np.sin(2 * np.pi * idx.hour / 24)
    noise = rng.normal(0, 30 + 40 * (df["hour"] > 16), len(idx))  # heteroscedastic by hour
    df["y_true"] = df["y_pred"] + noise
    df["timestamp"] = idx
    df["roll168_at168"] = 1000.0
    return df


def test_typed_casts_calendar_only():
    df = _typed(_frame("2019-01-01", 2, 0))
    for c in CATEGORICAL_FEATURES:
        assert str(df[c].dtype) == "category"
    assert all(df[c].dtype == float for c in CONTINUOUS_FEATURES)


def test_replica_matches_nannyml():
    nml = pytest.importorskip("nannyml")
    ref, ana = _frame("2019-01-01", 30, 1), _frame("2019-02-01", 10, 2)
    dle = nml.DLE(feature_column_names=FEATURES, y_pred="y_pred", y_true="y_true",
                  timestamp_column_name="timestamp", metrics=["mae", "rmse"],
                  chunk_period="D").fit(_typed(ref).reset_index(drop=True))
    est = dle.estimate(_typed(ana.drop(columns=["y_true"])).reset_index(drop=True)).to_df()
    est = est[est[("chunk", "period")] == "analysis"]
    rep = DLEReplica(FEATURES, CATEGORICAL_FEATURES).fit(ref)
    mine = DLEReplica.chunk(rep.row_estimates(ana), ana["timestamp"].dt.normalize())
    np.testing.assert_allclose(mine["mae"].to_numpy(), est[("mae", "value")].to_numpy(),
                               rtol=1e-6)
    np.testing.assert_allclose(mine["rmse"].to_numpy(), est[("rmse", "value")].to_numpy(),
                               rtol=1e-6)


def test_cross_fit_has_out_of_fold_reference_estimates():
    ref = _frame("2019-01-01", 20, 3)
    cf = DLEReplica(FEATURES, CATEGORICAL_FEATURES, cross_fit=True).fit(ref)
    assert len(cf.oof["mae"]) == len(ref) and len(cf.fold_models["mae"]) == 5
    full = DLEReplica(FEATURES, CATEGORICAL_FEATURES).fit(ref).row_estimates(ref)["mae"]
    oof = cf.oof["mae"]
    # in-sample fit hugs the noise more than out-of-fold estimates do
    err = np.abs(ref.y_true - ref.y_pred).to_numpy()
    assert np.corrcoef(full, err)[0, 1] > np.corrcoef(oof, err)[0, 1]


def test_input_drift_alert_count():
    t = pd.DataFrame({"uni_alert_rel_lag_168_wasserstein": [True, False],
                      "uni_alert_hour_jensen_shannon": [True, True],
                      "uni_alert_y_pred_wasserstein": [True, True]})
    assert input_drift_alerts(t).tolist() == [2, 1]  # y_pred is not an input
