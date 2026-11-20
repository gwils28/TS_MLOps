"""Shared pipeline steps: config, forecasts (cached), fitted monitor."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from w47.data import load_hourly
from w47.features import period_mask
from w47.model import add_predictions, build_frame, train_forecaster

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TABLES, FIGURES = RESULTS / "tables", RESULTS / "figures"


def load_config(path: str | Path = ROOT / "configs/protocol.yaml") -> dict:
    return yaml.safe_load(open(path))


@lru_cache(maxsize=1)
def series() -> pd.Series:
    return load_hourly(ROOT / load_config()["data"]["hourly_series"])


@lru_cache(maxsize=1)
def forecaster():
    """The frozen forecaster, trained once on the train period."""
    return train_forecaster(build_frame(series()), load_config())


@lru_cache(maxsize=1)
def forecasts() -> pd.DataFrame:
    """Features, target and predictions of the frozen forecaster on every hour."""
    return add_predictions(build_frame(series()), forecaster())


def slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[period_mask(df.index, start, end)]
