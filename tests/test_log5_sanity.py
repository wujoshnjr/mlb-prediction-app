"""
test_log5_sanity.py

Verify log5 probability sanity:
- no NaN/inf
- always in [0,1]
- neutral inputs produce 0.5
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


PREDICTION_PATH = Path("report/prediction.json")


def log5_prob(home_strength: Any, away_strength: Any) -> float:
    try:
        home = float(home_strength)
        away = float(away_strength)
    except (TypeError, ValueError):
        return 0.5

    if not math.isfinite(home) or not math.isfinite(away):
        return 0.5

    # Log5 inputs should be probabilities/strengths in [0,1].
    # Clamp only for this helper to ensure defensive behavior.
    home = max(0.0, min(1.0, home))
    away = max(0.0, min(1.0, away))

    denominator = home * (1.0 - away) + away * (1.0 - home)
    if denominator <= 0 or not math.isfinite(denominator):
        return 0.5

    probability = home * (1.0 - away) / denominator

    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        return 0.5

    return probability


def _load_predictions() -> List[Dict[str, Any]]:
    if not PREDICTION_PATH.exists():
        pytest.skip("report/prediction.json not found")

    with PREDICTION_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        predictions = data
    elif isinstance(data, dict):
        predictions = []
        for key in ("predictions", "today_predictions", "games"):
            value = data.get(key)
            if isinstance(value, list):
                predictions = value
                break
    else:
        predictions = []

    predictions = [item for item in predictions if isinstance(item, dict)]
    if not predictions:
        pytest.skip("no predictions found")

    return predictions


def _feature_value(prediction: Dict[str, Any], name: str) -> Optional[Any]:
    features = prediction.get("features")
    if isinstance(features, dict) and name in features:
        return features.get(name)
    return prediction.get(name)


def test_log5_formula_sanity() -> None:
    assert abs(log5_prob(0.5, 0.5) - 0.5) < 1e-9
    assert log5_prob(0.6, 0.5) > 0.5
    assert log5_prob(0.4, 0.5) < 0.5
    assert log5_prob(0.0, 0.0) == 0.5
    assert log5_prob(1.0, 1.0) == 0.5
    assert log5_prob(None, None) == 0.5


def test_prediction_log5_prob_is_valid_probability() -> None:
    predictions = _load_predictions()

    checked = 0
    for prediction in predictions:
        value = _feature_value(prediction, "log5_prob")
        if value is None:
            continue

        try:
            parsed = float(value)
        except (TypeError, ValueError):
            pytest.fail(f"log5_prob is not numeric: {value}")

        checked += 1
        assert math.isfinite(parsed), f"log5_prob is not finite: {parsed}"
        assert 0.0 <= parsed <= 1.0, f"log5_prob out of range: {parsed}"

    if checked == 0:
        pytest.skip("no log5_prob in predictions")
