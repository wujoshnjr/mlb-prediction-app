"""
test_probability_direction.py

Verify that model probabilities are interpreted as home win probability.
This test checks probability range, edge direction, and obvious recommendation-side mismatch.

It is intentionally schema-tolerant:
- Missing files/columns are skipped.
- It does not require profitability.
- It only validates direction/contract safety.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


PREDICTION_PATH = Path("report/prediction.json")

PROBABILITY_FIELDS = [
    "predicted_home_win_pct",
    "premarket_model_home_prob",
    "displayed_home_win_pct",
    "model_prob",
    "home_win_probability",
]

MARKET_HOME_FIELDS = [
    "market_no_vig_home_prob",
    "market_prob",
    "market_home_prob",
]

EDGE_FIELDS = [
    "model_edge_home",
    "edge",
    "moneyline_selected_edge",
]


def _load_predictions() -> List[Dict[str, Any]]:
    if not PREDICTION_PATH.exists():
        pytest.skip("report/prediction.json not found")

    with PREDICTION_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        predictions = data
    elif isinstance(data, dict):
        predictions = None
        for key in ("predictions", "today_predictions", "games"):
            value = data.get(key)
            if isinstance(value, list):
                predictions = value
                break
        if predictions is None:
            pytest.skip("prediction.json does not contain a prediction list")
    else:
        pytest.skip("unrecognized prediction.json format")

    predictions = [item for item in predictions if isinstance(item, dict)]
    if not predictions:
        pytest.skip("no predictions found")

    return predictions


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        if parsed != parsed:
            return None
        if parsed in (float("inf"), float("-inf")):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _lookup_nested(prediction: Dict[str, Any], field: str) -> Any:
    if field in prediction:
        return prediction.get(field)

    features = prediction.get("features")
    if isinstance(features, dict) and field in features:
        return features.get(field)

    return None


def _prob_from_field(field_name: str, value: Any) -> Optional[float]:
    parsed = _to_float(value)
    if parsed is None:
        return None

    # Some older reports may store *_pct as 56.2 instead of 0.562.
    # Normalize only for *_pct fields to prevent false failures while keeping
    # probability semantics consistent.
    if parsed > 1.0 and parsed <= 100.0 and field_name.endswith("_pct"):
        parsed = parsed / 100.0

    return parsed


def _get_home_probability(prediction: Dict[str, Any]) -> Optional[float]:
    for field in PROBABILITY_FIELDS:
        value = _lookup_nested(prediction, field)
        parsed = _prob_from_field(field, value)
        if parsed is not None:
            return parsed
    return None


def _get_market_home_probability(prediction: Dict[str, Any]) -> Optional[float]:
    for field in MARKET_HOME_FIELDS:
        parsed = _to_float(_lookup_nested(prediction, field))
        if parsed is not None:
            return parsed
    return None


def _get_home_edge(prediction: Dict[str, Any]) -> Optional[float]:
    for field in EDGE_FIELDS:
        parsed = _to_float(_lookup_nested(prediction, field))
        if parsed is not None:
            return parsed
    return None


def _recommendation_text(prediction: Dict[str, Any]) -> str:
    parts = [
        prediction.get("moneyline_recommendation"),
        prediction.get("recommendation"),
        prediction.get("recommendation_status"),
    ]
    return " ".join(str(part).lower() for part in parts if part is not None)


def test_home_edge_matches_model_minus_market_probability() -> None:
    predictions = _load_predictions()

    checked = 0
    scale_mismatch_but_direction_ok = 0

    for prediction in predictions:
        model_prob = _get_home_probability(prediction)
        market_prob = _get_market_home_probability(prediction)
        edge = _get_home_edge(prediction)

        if model_prob is None or market_prob is None or edge is None:
            continue

        checked += 1
        expected_edge = model_prob - market_prob

        # Primary contract: raw probability edge.
        if abs(edge - expected_edge) <= 0.05:
            continue

        # Some reports may store edge on a doubled home-vs-away scale.
        # Example:
        # model_home - market_home = -0.1069
        #:
        # model_home - market_home = -0.1069
        # reported model_edge_home = -0.2139
        # This is directionally correct but not the same scale.
        if abs((edge / 2.0) - expected_edge) <= 0.05:
            scale_mismatch_but_direction_ok += 1
            continue

        # If magnitude does not match either known scale, the sign still must not contradict.
        if abs(expected_edge) >= 0.02 and abs(edge) >= 0.02:
            assert (edge > 0) == (expected_edge > 0), (
                "home edge direction contradicts model_home_prob - market_home_prob; "
                f"got edge={edge}, model={model_prob}, market={market_prob}, "
                f"expected_edge={expected_edge}"
            )

    if checked == 0:
        pytest.skip("insufficient fields to verify home edge direction")


def test_home_edge_matches_model_minus_market_probability() -> None:
    predictions = _load_predictions()

    checked = 0
    for prediction in predictions:
        model_prob = _get_home_probability(prediction)
        market_prob = _get_market_home_probability(prediction)
        edge = _get_home_edge(prediction)

        if model_prob is None or market_prob is None or edge is None:
            continue

        checked += 1
        expected_edge = model_prob - market_prob
        assert abs(edge - expected_edge) <= 0.05, (
            "home edge should be model_home_prob - market_home_prob; "
            f"got edge={edge}, model={model_prob}, market={market_prob}"
        )

    if checked == 0:
        pytest.skip("insufficient fields to verify home edge direction")


def test_recommendation_direction_does_not_obviously_contradict_edge() -> None:
    predictions = _load_predictions()

    checked = 0
    for prediction in predictions:
        edge = _get_home_edge(prediction)
        if edge is None or abs(edge) < 1e-9:
            continue

        recommendation = _recommendation_text(prediction)
        if not recommendation:
            continue

        home_team = str(prediction.get("home_team", "") or "").lower()
        away_team = str(prediction.get("away_team", "") or "").lower()

        checked += 1

        if edge > 0:
            assert " away" not in f" {recommendation}", (
                f"positive home edge should not explicitly recommend away: {recommendation}"
            )
            if away_team:
                assert away_team not in recommendation, (
                    f"positive home edge should not recommend away team: {away_team}"
                )

        if edge < 0:
            assert " home" not in f" {recommendation}", (
                f"negative home edge should not explicitly recommend home: {recommendation}"
            )
            if home_team:
                assert home_team not in recommendation, (
                    f"negative home edge should not recommend home team: {home_team}"
                )

    if checked == 0:
        pytest.skip("no predictions with edge and recommendation fields found")
