"""
test_no_vig_mapping_contract.py

Verify that no-vig odds mapping represents home no-vig probability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest


SNAPSHOTS_PATH = Path("data/prediction_snapshots.csv")


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        if parsed != parsed:
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def no_vig_home_probability(home_odds: Any, away_odds: Any) -> Optional[float]:
    home = _to_float(home_odds)
    away = _to_float(away_odds)

    if home is None or away is None:
        return None

    if home <= 1.0 or away <= 1.0:
        return None

    home_implied = 1.0 / home
    away_implied = 1.0 / away
    total = home_implied + away_implied

    if total <= 0:
        return None

    return home_implied / total


def test_no_vig_formula_direction() -> None:
    assert no_vig_home_probability(2.0, 2.0) == pytest.approx(0.5)
    assert no_vig_home_probability(1.8, 2.2) > 0.5
    assert no_vig_home_probability(2.2, 1.8) < 0.5


def test_snapshot_market_no_vig_home_probability_mapping() -> None:
    if not SNAPSHOTS_PATH.exists():
        pytest.skip("data/prediction_snapshots.csv not found")

    frame = pd.read_csv(SNAPSHOTS_PATH)
    required = {"home_moneyline_odds", "away_moneyline_odds", "market_no_vig_home_prob"}
    missing = required - set(frame.columns)
    if missing:
        pytest.skip(f"missing required columns: {sorted(missing)}")

    checked = 0
    mismatches = []

    for index, row in frame.iterrows():
        expected = no_vig_home_probability(
            row.get("home_moneyline_odds"),
            row.get("away_moneyline_odds"),
        )
        actual = _to_float(row.get("market_no_vig_home_prob"))

        if expected is None or actual is None:
            continue

        checked += 1
        if abs(expected - actual) > 0.02:
            mismatches.append(
                {
                    "index": int(index),
                    "game_id": row.get("game_id"),
                    "expected_home_no_vig": expected,
                    "actual_market_no_vig_home_prob": actual,
                }
            )

    if checked == 0:
        pytest.skip("no valid snapshot rows with odds and market_no_vig_home_prob")

    assert not mismatches, f"no-vig home probability mapping mismatch: {mismatches[:5]}"
