"""Monitoring layer: NannyML calculators + a transparent DLE replica, flattened to one tidy table.

Output of ``Monitor.run``: one row per daily chunk (target day) with

* ``est_mae`` / ``est_rmse``: NannyML DLE estimates (computed **without** ``y_true``),
  their sampling error, thresholds and alert flags;
* ``real_mae`` / ``real_rmse``: realized performance (needs ``y_true``), thresholds, alerts;
* ``uni_<feature>_<method>`` values and ``uni_alert_<feature>_<method>`` flags;
* ``pca_err`` / ``pca_alert``: multivariate drift (PCA reconstruction error);
* ``support_out``: share of rows with a continuous input (or ``y_pred``) outside the
  reference [min, max];
* ``rep_mae`` / ``cf_mae``: our DLE replica (in-sample, like NannyML) and its cross-fitted
  variant (nanny model fitted out of fold on the reference).
"""

from __future__ import annotations

import os

os.environ.setdefault("NML_DISABLE_USAGE_LOGGING", "1")  # NannyML phones home by default

import lightgbm as lgb  # noqa: E402
import nannyml as nml  # noqa: E402
import numpy as np  # noqa: E402
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


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    """NannyML decides categorical vs continuous from the dtype (``_split_features_by_type``):
    int calendar columns would silently be treated as continuous by DLE and PCA. Cast them."""
    out = df.copy()
    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].astype("category")
    return out


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


class DLEReplica:
    """Direct Loss Estimation in a few lines, to make the mechanism explicit.

    NannyML 0.13 (read in ``direct_loss_estimation/metrics.py``): for MAE the nanny model is a
    default ``LGBMRegressor()`` fitted on reference rows, inputs = features + y_pred, target =
    |y - y_pred|; the chunk estimate is the mean of the clipped row predictions. For RMSE the
    target is (y - y_pred)^2 and the chunk estimate is sqrt(mean). ``cross_fit=True`` fits the
    nanny model out of fold (k contiguous folds) so that reference estimates are honest.
    """

    def __init__(self, features: list[str], categorical: list[str], cross_fit: bool = False,
                 n_folds: int = 5, params: dict | None = None):
        self.features, self.categorical = features, categorical
        self.cross_fit, self.n_folds = cross_fit, n_folds
        self.params = params or {}

    def _inputs(self, df):
        return df[self.features + ["y_pred"]]

    def _new(self):
        return lgb.LGBMRegressor(verbose=-1, **self.params)

    def fit(self, ref: pd.DataFrame) -> DLEReplica:
        X = self._inputs(ref)
        err = ref["y_true"].to_numpy() - ref["y_pred"].to_numpy()
        self.targets = {"mae": np.abs(err), "rmse": err**2}
        self.models = {
            k: self._new().fit(X, t, categorical_feature=self.categorical)
            for k, t in self.targets.items()
        }
        if self.cross_fit:  # out-of-fold row estimates on the reference itself
            folds = np.array_split(np.arange(len(ref)), self.n_folds)
            self.oof = {k: np.zeros(len(ref)) for k in self.targets}
            self.fold_models = {k: [] for k in self.targets}
            for idx in folds:
                train = np.setdiff1d(np.arange(len(ref)), idx)
                for k, t in self.targets.items():
                    m = self._new().fit(X.iloc[train], t[train],
                                        categorical_feature=self.categorical)
                    self.oof[k][idx] = np.maximum(0, m.predict(X.iloc[idx]))
                    self.fold_models[k].append(m)
        return self

    def row_estimates(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """In-sample variant: the full-reference model. Cross-fitted variant: the average of the
        fold models (each never saw one fifth of the reference)."""
        X = self._inputs(df)
        if self.cross_fit:
            return {k: np.mean([np.maximum(0, m.predict(X)) for m in ms], axis=0)
                    for k, ms in self.fold_models.items()}
        return {k: np.maximum(0, m.predict(X)) for k, m in self.models.items()}

    @staticmethod
    def chunk(row_est: dict[str, np.ndarray], days: pd.Series) -> pd.DataFrame:
        tmp = pd.DataFrame({"day": days.to_numpy(), "mae": row_est["mae"],
                            "mse": row_est["rmse"]})
        g = tmp.groupby("day")
        return pd.DataFrame({"mae": g["mae"].mean(), "rmse": np.sqrt(g["mse"].mean())})


class Monitor:
    """Fit every monitor on the reference, then run them on any analysis frame."""

    def __init__(self, chunk_period: str = "D", metrics=("mae", "rmse"),
                 continuous_methods=("kolmogorov_smirnov", "wasserstein"),
                 categorical_methods=("chi2", "jensen_shannon"), with_univariate: bool = True,
                 with_replica: bool = True):
        self.chunk_period = chunk_period
        self.metrics = list(metrics)
        self.cont_methods, self.cat_methods = list(continuous_methods), list(categorical_methods)
        self.with_univariate, self.with_replica = with_univariate, with_replica

    def fit(self, ref: pd.DataFrame) -> Monitor:
        ref = ref.reset_index(drop=True)
        tref = _typed(ref)
        kw = dict(timestamp_column_name="timestamp", chunk_period=self.chunk_period)
        self.dle = nml.DLE(feature_column_names=FEATURES, y_pred="y_pred", y_true="y_true",
                           metrics=self.metrics, **kw).fit(tref)
        self.perf = nml.PerformanceCalculator(problem_type="regression", y_pred="y_pred",
                                              y_true="y_true", metrics=self.metrics,
                                              **kw).fit(tref)
        self.pca = nml.DataReconstructionDriftCalculator(column_names=FEATURES, **kw).fit(tref)
        if self.with_univariate:
            self.uni = nml.UnivariateDriftCalculator(
                column_names=UNI_CONTINUOUS + UNI_CATEGORICAL,
                treat_as_categorical=UNI_CATEGORICAL,
                continuous_methods=self.cont_methods, categorical_methods=self.cat_methods,
                **kw).fit(tref)
        self.ref_min = ref[SUPPORT_COLUMNS].min()
        self.ref_max = ref[SUPPORT_COLUMNS].max()
        if self.with_replica:
            self.rep = DLEReplica(FEATURES, CATEGORICAL_FEATURES).fit(ref)
            self.cf = DLEReplica(FEATURES, CATEGORICAL_FEATURES, cross_fit=True).fit(ref)
            self.cf_ref = DLEReplica.chunk(
                {"mae": self.cf.oof["mae"], "rmse": self.cf.oof["rmse"]},
                ref["timestamp"].dt.normalize())
            self.rep_ref = DLEReplica.chunk(self.rep.row_estimates(ref),
                                            ref["timestamp"].dt.normalize())
            self.real_ref = ref.assign(day=ref["timestamp"].dt.normalize(),
                                       ae=(ref.y_true - ref.y_pred).abs(),
                                       se=(ref.y_true - ref.y_pred) ** 2).groupby("day").agg(
                mae=("ae", "mean"), mse=("se", "mean"))
            self.real_ref["rmse"] = np.sqrt(self.real_ref.pop("mse"))
        return self

    def run(self, ana: pd.DataFrame, labels: bool = True) -> pd.DataFrame:
        ana = ana.reset_index(drop=True)
        blind = ana.drop(columns=["y_true"])  # DLE and drift never see the labels
        tblind = _typed(blind)
        est = _flat(self.dle.estimate(tblind).to_df())
        out = pd.DataFrame(index=est.index)
        for m in self.metrics:
            out[f"est_{m}"] = est[(m, "value")].astype(float)
            out[f"est_{m}_se"] = est[(m, "sampling_error")].astype(float)
            out[f"{m}_upper"] = est[(m, "upper_threshold")].astype(float)
            out[f"{m}_lower"] = est[(m, "lower_threshold")].astype(float)
            out[f"est_{m}_alert"] = est[(m, "alert")].astype(bool)
        if labels:
            real = _flat(self.perf.calculate(_typed(ana)).to_df())
            for m in self.metrics:
                out[f"real_{m}"] = real[(m, "value")].astype(float).reindex(out.index)
                out[f"real_{m}_alert"] = real[(m, "alert")].astype(bool).reindex(out.index)
        pca = _flat(self.pca.calculate(tblind).to_df())
        out["pca_err"] = pca[("reconstruction_error", "value")].astype(float).reindex(out.index)
        out["pca_upper"] = pca[("reconstruction_error", "upper_threshold")].astype(float).reindex(
            out.index)
        out["pca_alert"] = pca[("reconstruction_error", "alert")].astype(bool).reindex(out.index)
        if self.with_univariate:
            uni = _flat(self.uni.calculate(tblind).to_df())
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
        if self.with_replica:
            rep = DLEReplica.chunk(self.rep.row_estimates(blind), day)
            out["rep_mae"], out["rep_rmse"] = rep["mae"], rep["rmse"]
            cf = DLEReplica.chunk(self.cf.row_estimates(blind), day)
            out["cf_mae"], out["cf_rmse"] = cf["mae"], cf["rmse"]
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
