"""Monitoring layer: NannyML calculators flattened to one tidy table per daily chunk.

Output of ``Monitor.run``: one row per daily chunk (target day) with

* ``est_mae`` / ``est_rmse``: NannyML DLE estimates (computed **without** ``y_true``),
  their sampling error, thresholds and alert flags;
* ``real_mae`` / ``real_rmse``: realized performance (needs ``y_true``), thresholds, alerts;
* ``uni_<feature>_<method>`` values and ``uni_alert_<feature>_<method>`` flags;
* ``pca_err`` / ``pca_alert``: multivariate drift (PCA reconstruction error);
* ``support_out``: share of rows with a continuous input (or ``y_pred``) outside the
  reference [min, max];
"""

from __future__ import annotations

import os

os.environ.setdefault("NML_DISABLE_USAGE_LOGGING", "1")  # NannyML phones home by default

import nannyml as nml  # noqa: E402
import pandas as pd  # noqa: E402

from w47.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURES,
    LEVEL,
)

UNI_CONTINUOUS = CONTINUOUS_FEATURES + [LEVEL, "y_pred"]
UNI_CATEGORICAL = CATEGORICAL_FEATURES
INPUT_DRIFT_COLUMNS = CONTINUOUS_FEATURES + [LEVEL] + CATEGORICAL_FEATURES  # excludes y_pred
SUPPORT_COLUMNS = CONTINUOUS_FEATURES + ["y_pred"]


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    """Keep analysis chunks only and index them by chunk start date."""
    if isinstance(df.columns, pd.MultiIndex):
        period = df[[c for c in df.columns if c[-1] == "period"][0]]
        start = df[[c for c in df.columns if c[-1] == "start_date"][0]]
    else:
        period, start = df["period"], df["start_date"]
    out = df[period.to_numpy() == "analysis"].copy()
    out.index = pd.to_datetime(start[period.to_numpy() == "analysis"]).dt.normalize()
    out.index.name = "day"
    return out


class Monitor:
    """Fit every monitor on the reference, then run them on any analysis frame."""

    def __init__(self, chunk_period: str = "D", metrics=("mae", "rmse"),
                 continuous_methods=("kolmogorov_smirnov", "wasserstein"),
                 categorical_methods=("chi2", "jensen_shannon"), with_univariate: bool = True):
        self.chunk_period = chunk_period
        self.metrics = list(metrics)
        self.cont_methods, self.cat_methods = list(continuous_methods), list(categorical_methods)
        self.with_univariate = with_univariate

    def fit(self, ref: pd.DataFrame) -> Monitor:
        ref = ref.reset_index(drop=True)
        kw = dict(timestamp_column_name="timestamp", chunk_period=self.chunk_period)
        self.dle = nml.DLE(feature_column_names=FEATURES, y_pred="y_pred", y_true="y_true",
                           metrics=self.metrics, **kw).fit(ref)
        self.perf = nml.PerformanceCalculator(problem_type="regression", y_pred="y_pred",
                                              y_true="y_true", metrics=self.metrics,
                                              **kw).fit(ref)
        self.pca = nml.DataReconstructionDriftCalculator(column_names=FEATURES, **kw).fit(ref)
        if self.with_univariate:
            self.uni = nml.UnivariateDriftCalculator(
                column_names=UNI_CONTINUOUS + UNI_CATEGORICAL,
                treat_as_categorical=UNI_CATEGORICAL,
                continuous_methods=self.cont_methods, categorical_methods=self.cat_methods,
                **kw).fit(ref)
        self.ref_min = ref[SUPPORT_COLUMNS].min()
        self.ref_max = ref[SUPPORT_COLUMNS].max()
        return self

    def run(self, ana: pd.DataFrame, labels: bool = True) -> pd.DataFrame:
        ana = ana.reset_index(drop=True)
        blind = ana.drop(columns=["y_true"])  # DLE and drift never see the labels
        est = _flat(self.dle.estimate(blind).to_df())
        out = pd.DataFrame(index=est.index)
        for m in self.metrics:
            out[f"est_{m}"] = est[(m, "value")].astype(float)
            out[f"est_{m}_se"] = est[(m, "sampling_error")].astype(float)
            out[f"{m}_upper"] = est[(m, "upper_threshold")].astype(float)
            out[f"{m}_lower"] = est[(m, "lower_threshold")].astype(float)
            out[f"est_{m}_alert"] = est[(m, "alert")].astype(bool)
        if labels:
            real = _flat(self.perf.calculate(ana).to_df())
            for m in self.metrics:
                out[f"real_{m}"] = real[(m, "value")].astype(float).reindex(out.index)
                out[f"real_{m}_alert"] = real[(m, "alert")].astype(bool).reindex(out.index)
        pca = _flat(self.pca.calculate(blind).to_df())
        out["pca_err"] = pca[("reconstruction_error", "value")].astype(float).reindex(out.index)
        out["pca_upper"] = pca[("reconstruction_error", "upper_threshold")].astype(float).reindex(
            out.index)
        out["pca_alert"] = pca[("reconstruction_error", "alert")].astype(bool).reindex(out.index)
        if self.with_univariate:
            uni = _flat(self.uni.calculate(blind).to_df())
            for col, method, field in uni.columns:
                if col == "chunk" or field not in ("value", "alert"):
                    continue
                key = f"uni_{col}_{method}" if field == "value" else f"uni_alert_{col}_{method}"
                out[key] = uni[(col, method, field)].reindex(out.index)
        day = ana["timestamp"].dt.normalize()
        outside = ((ana[SUPPORT_COLUMNS] < self.ref_min) | (ana[SUPPORT_COLUMNS] > self.ref_max))
        out["support_out"] = outside.any(axis=1).groupby(day.to_numpy()).mean().reindex(out.index)
        out["support_out_ypred"] = outside["y_pred"].groupby(day.to_numpy()).mean().reindex(
            out.index)
        return out


def input_drift_alerts(table: pd.DataFrame, methods=("wasserstein", "jensen_shannon"),
                       columns=None) -> pd.Series:
    """Number of input columns (y_pred excluded) whose univariate drift alerts, per chunk."""
    columns = columns or INPUT_DRIFT_COLUMNS
    keys = [f"uni_alert_{c}_{m}" for c in columns for m in methods
            if f"uni_alert_{c}_{m}" in table.columns]
    per_col = {}
    for c in columns:
        ks = [k for k in keys if k.startswith(f"uni_alert_{c}_")]
        if ks:
            per_col[c] = table[ks].fillna(False).astype(bool).any(axis=1)
    return pd.DataFrame(per_col).sum(axis=1)
