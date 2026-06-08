from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_prediction_trust_report import build_report


def test_missing_prediction_report_does_not_crash(tmp_path: Path) -> None:
    report = build_report(
        prediction_path=tmp_path / "missing.json",
        output_path=tmp_path / "report" / "prediction_trust_report.json",
    )

    assert report["status"] == "partial"
    assert report["prediction_count"] == 0


def test_complete_data_can_get_a_or_b(tmp_path: Path) -> None:
    prediction = {
        "games": [
            {
                "game_id": "1",
                "home_team": "A",
                "away_team": "B",
                "model_probability": 0.58,
                "market_probability": 0.55,
                "lineup_status": "confirmed",
                "starter_status": "confirmed",
                "odds_quality_status": "ok",
                "data_quality_grade": "A",
            }
        ]
    }

    pp = tmp_path / "prediction.json"
    pp.write_text(json.dumps(prediction), encoding="utf-8")

    report = build_report(
        prediction_path=pp,
        output_path=tmp_path / "report" / "prediction_trust_report.json",
    )

    assert report["prediction_count"] == 1
    assert report["predictions"][0]["trust_grade"] in {"A", "B"}


def test_missing_odds_and_starter_downgrade(tmp_path: Path) -> None:
    prediction = {
        "games": [
            {
                "game_id": "1",
                "model_probability": 0.58,
                "lineup_status": "unconfirmed",
                "starter_status": "missing",
                "odds_quality_status": "missing",
                "data_quality_grade": "C",
            }
        ]
    }

    pp = tmp_path / "prediction.json"
    pp.write_text(json.dumps(prediction), encoding="utf-8")

    report = build_report(
        prediction_path=pp,
        output_path=tmp_path / "report" / "prediction_trust_report.json",
    )

    assert report["predictions"][0]["trust_grade"] in {"C", "D"}
