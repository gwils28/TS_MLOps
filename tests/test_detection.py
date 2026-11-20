"""Timing model of the detection bench: the two clocks."""

import numpy as np
import pandas as pd

from w47.e3_detection import detector_flags, evaluate

THR = {"d3": 100.0, "d2": 90.0, "w3": 100.0, "w2": 90.0, "rel_real": 100.0, "rel_est": 100.0}


def _run(onset_idx=14, jump_real=True, jump_est=False):
    days = pd.date_range("2019-08-01", periods=42, freq="D")
    rel = np.arange(42) - onset_idx
    t = pd.DataFrame(index=days)
    t["rel_day"] = rel
    t["onset"] = days[onset_idx]
    t["real_mae"] = np.where((rel >= 0) & jump_real, 500.0, 50.0)
    t["est_mae"] = np.where((rel >= 0) & jump_est, 500.0, 50.0)
    t["mae_upper"] = 100.0
    t["n_input_drift"] = np.where((rel >= 0) & jump_est, 3, 0)
    t["pca_alert"] = False
    t["scenario"], t["family"], t["run"] = "x", "cd_bias", 0
    return t


def test_realized_clock_is_target_plus_one_plus_latency(monkeypatch):
    monkeypatch.setattr("w47.e3_detection.reference_thresholds", lambda: THR)
    ev = evaluate(_run(), k=1, latency_days=7)
    r = ev[ev.detector == "réalisé (labels)"].iloc[0]
    assert r.target_delay == 0 and r.clock_delay == 8


def test_label_free_clock_is_issue_time(monkeypatch):
    monkeypatch.setattr("w47.e3_detection.reference_thresholds", lambda: THR)
    ev = evaluate(_run(jump_real=False, jump_est=True), k=1, latency_days=0)
    r = ev[ev.detector == "DLE seul"].iloc[0]
    assert r.target_delay == 0 and r.clock_delay == -6  # visible one week ahead, at issue time


def test_concept_drift_is_invisible_to_label_free_detectors():
    f = detector_flags(_run(jump_real=True, jump_est=False), k=1, thr=THR)
    assert not f["DLE seul"].any() and not f["règle jointe (dérive ET DLE)"].any()
    assert f["réalisé (labels)"].any()
