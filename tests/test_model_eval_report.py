# tests/test_model_eval_report.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import model_eval_report
from scripts.feature_schema import CORE_MODEL_FEATURES


def make_training_frame(row_count: int) -> pd.DataFrame:
    dates = pd.date_range("2025-04-01", periods=row_count, freq="D")
    labels = np.array([index % 2 for index in range(row_count)], dtype=int)

    frame = pd.DataFrame(
        {
            "game_id": [f"game_{index}" for index in range(row_count)],
            "game_date": dates.strftime("%Y-%m-%d"),
            "home_win": labels,
            "home_team": ["HOME"] * row_count,
            "away_team": ["AWAY"] * row_count,
            "selected_side": ["home" if value == 1 else "away" for value in labels],
            "edge_bucket": ["below_threshold"] * row_count,
            "odds_quality_status": ["OK"] * row_count,
            "model_source": ["eval"] * row_count,
        }
    )

    for feature_index, feature in enumerate(CORE_MODEL_FEATURES):
        frame[feature] = (
            labels * (feature_index + 1) * 0.1
            + np.linspace(0.0, 1.0, row_count)
            + feature_index
        )

    return frame


def test_missing_training_csv_skips(tmp_path, monkeypatch):
    report_path = tmp_path / "model_eval_report.json"
    oos_path = tmp_path / "oos_predictions_with_labels.csv"

    monkeypatch.setattr(model_eval_report, "DATA_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(model_eval_report, "REPORT_JSON_PATH", report_path)
    monkeypatch.setattr(model_eval_report, "OOS_CSV_PATH", oos_path)

    report = model_eval_report.generate_report()

    assert report["status"] == "skipped"
    assert report_path.exists()

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["live_betting_allowed"] is False
    assert data["automated_wagering_allowed"] is False
    assert data["production_model_replacement_allowed"] is False


def test_small_sample_skips(tmp_path, monkeypatch):
    data_path = tmp_path / "training_samples.csv"
    report_path = tmp_path / "model_eval_report.json"
    oos_path = tmp_path / "oos_predictions_with_labels.csv"

    make_training_frame(12).to_csv(data_path, index=False)

    monkeypatch.setattr(model_eval_report, "DATA_PATH", data_path)
    monkeypatch.setattr(model_eval_report, "REPORT_JSON_PATH", report_path)
    monkeypatch.setattr(model_eval_report, "OOS_CSV_PATH", oos_path)

    report = model_eval_report.generate_report()

    assert report["status"] == "skipped"
    assert "insufficient_samples" in ";".join(report["errors"])


def test_model_eval_report_success_with_synthetic_data(tmp_path, monkeypatch):
    data_path = tmp_path / "training_samples.csv"
    report_path = tmp_path / "model_eval_report.json"
    oos_path = tmp_path / "oos_predictions_with_labels.csv"

    make_training_frame(80).to_csv(data_path, index=False)

    monkeypatch.setattr(model_eval_report, "DATA_PATH", data_path)
    monkeypatch.setattr(model_eval_report, "REPORT_JSON_PATH", report_path)
    monkeypatch.setattr(model_eval_report, "OOS_CSV_PATH", oos_path)

    report = model_eval_report.generate_report()

    assert report["status"] == "ok"
    assert oos_path.exists()
    assert report["test_sample_count"] > 0
    assert "accuracy" in report["metrics"]
    assert report["live_betting_allowed"] is False
    assert report["automated_wagering_allowed"] is False
    assert report["production_model_replacement_allowed"] is False

    payload = report_path.read_text(encoding="utf-8")
    assert "NaN" not in payload
    assert "Infinity" not in payload
    assert "-Infinity" not in payload


def test_clean_json_value_removes_nan_and_inf():
    cleaned = model_eval_report.clean_json_value(
        {
            "nan": float("nan"),
            "inf": float("inf"),
            "neg_inf": float("-inf"),
            "np_nan": np.float64(np.nan),
            "ok": 1.25,
        }
    )

    payload = json.dumps(cleaned, allow_nan=False)

    assert "NaN" not in payload
    assert "Infinity" not in payload
    assert cleaned["nan"] is None
    assert cleaned["inf"] is None
    assert cleaned["neg_inf"] is None
    assert cleaned["np_nan"] is None
