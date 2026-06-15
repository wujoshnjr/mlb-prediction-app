# tests/test_calibration_and_slice_reports.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibration_report
from scripts import per_slice_performance_report


@pytest.fixture
def temp_oos(tmp_path: Path) -> Path:
    row_count = 50
    labels = np.array([index % 2 for index in range(row_count)], dtype=int)
    probabilities = np.linspace(0.15, 0.85, row_count)

    frame = pd.DataFrame(
        {
            "game_id": [f"game_{index}" for index in range(row_count)],
            "game_date": pd.date_range("2025-06-01", periods=row_count).strftime("%Y-%m-%d"),
            "y_true": labels,
            "y_prob": probabilities,
            "home_team": ["HOU"] * 10 + ["LAD"] * 10 + ["NYY"] * 30,
            "away_team": ["BOS"] * 20 + ["CHC"] * 30,
            "selected_side": ["home" if value == 1 else "away" for value in labels],
            "edge_bucket": ["low", "mid", "high", "low", "mid"] * 10,
            "odds_quality_status": ["OK"] * row_count,
            "model_source": ["eval"] * row_count,
        }
    )

    path = tmp_path / "oos_predictions_with_labels.csv"
    frame.to_csv(path, index=False)
    return path


def test_calibration_skipped_when_missing(tmp_path, monkeypatch):
    report_path = tmp_path / "calibration_report.json"

    monkeypatch.setattr(calibration_report, "OOS_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(calibration_report, "REPORT_PATH", report_path)

    report = calibration_report.generate_report()

    assert report["status"] == "skipped"
    assert report_path.exists()


def test_calibration_with_data(temp_oos, tmp_path, monkeypatch):
    report_path = tmp_path / "calibration_report.json"

    monkeypatch.setattr(calibration_report, "OOS_PATH", temp_oos)
    monkeypatch.setattr(calibration_report, "REPORT_PATH", report_path)

    report = calibration_report.generate_report()

    assert report["status"] == "ok"
    assert 0.0 <= report["ece"] <= 1.0
    assert 0.0 <= report["mce"] <= 1.0
    assert len(report["reliability_table"]) == calibration_report.N_BINS
    assert report["live_betting_allowed"] is False
    assert report["automated_wagering_allowed"] is False
    assert report["production_model_replacement_allowed"] is False

    payload = report_path.read_text(encoding="utf-8")
    assert "NaN" not in payload
    assert "Infinity" not in payload
    assert "-Infinity" not in payload


def test_slice_report_skipped_when_missing(tmp_path, monkeypatch):
    report_path = tmp_path / "per_slice_performance_report.json"

    monkeypatch.setattr(per_slice_performance_report, "OOS_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(per_slice_performance_report, "REPORT_PATH", report_path)

    report = per_slice_performance_report.generate_report()

    assert report["status"] == "skipped"
    assert report_path.exists()


def test_slice_report_format(temp_oos, tmp_path, monkeypatch):
    report_path = tmp_path / "per_slice_performance_report.json"

    monkeypatch.setattr(per_slice_performance_report, "OOS_PATH", temp_oos)
    monkeypatch.setattr(per_slice_performance_report, "REPORT_PATH", report_path)

    report = per_slice_performance_report.generate_report()

    assert report["status"] in {"ok", "warning"}
    assert "slices" in report
    assert report["live_betting_allowed"] is False
    assert report["automated_wagering_allowed"] is False
    assert report["production_model_replacement_allowed"] is False

    for section in report["slices"].values():
        if section.get("status") != "ok":
            continue
        for entry in section["entries"]:
            assert "count" in entry
            assert "accuracy" in entry
            assert "brier" in entry

    payload = report_path.read_text(encoding="utf-8")
    assert "NaN" not in payload
    assert "Infinity" not in payload
    assert "-Infinity" not in payload
