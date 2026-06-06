from __future__ import annotations

import json
from pathlib import Path

import pytest


PREDICTION_JSON = Path("report/prediction.json")

REQUIRED_FIELDS = {
    "game_id",
    "model_governance_status",
    "data_quality_status",
    "live_betting_allowed",
    "live_bet_candidate",
    "stake_multiplier",
    "recommendation",
    "features",
}


def _predictions(report: object) -> list[dict]:
    if isinstance(report, list):
        return [item for item in report if isinstance(item, dict)]
    if isinstance(report, dict):
        raw = report.get("predictions") or report.get("today_predictions") or report.get("games") or []
        return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    return []


def test_prediction_contract_fields_and_live_lock() -> None:
    if not PREDICTION_JSON.exists():
        pytest.skip("prediction.json not found")

    data = json.loads(PREDICTION_JSON.read_text(encoding="utf-8"))
    predictions = _predictions(data)

    if not predictions:
        pytest.skip("prediction.json has no predictions")

    for index, prediction in enumerate(predictions):
        missing = REQUIRED_FIELDS - set(prediction)
        assert not missing, f"Prediction {index} missing fields: {sorted(missing)}"

        if prediction.get("live_betting_allowed") is False:
            assert prediction.get("live_bet_candidate") is False
            assert float(prediction.get("stake_multiplier") or 0.0) == 0.0
