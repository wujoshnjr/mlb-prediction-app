"""
test_outcome_label_contract.py

Verify that finalized outcome labels are not reversed.

Contract:
- home_win = 1 means home_score > away_score
- home_win = 0 means home_score < away_score
- outcome, when present, must agree with home_win
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest


FINALIZED_PATH = Path("data/finalized_games.csv")


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


def _to_binary(value: Any) -> Optional[int]:
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "home", "h", "home_win"}:
        return 1
    if text in {"0", "0.0", "false", "away", "a", "away_win"}:
        return 0

    parsed = _to_float(value)
    if parsed is None:
        return None
    if parsed == 1:
        return 1
    if parsed == 0:
        return 0

    return None


def _load_finalized() -> pd.DataFrame:
    if not FINALIZED_PATH.exists():
        pytest.skip("data/finalized_games.csv not found")

    frame = pd.read_csv(FINALIZED_PATH)
    if frame.empty:
        pytest.skip("finalized_games.csv is empty")

    return frame


def test_home_win_matches_scores() -> None:
    frame = _load_finalized()

    required = {"home_score", "away_score", "home_win"}
    missing = required - set(frame.columns)
    if missing:
        pytest.skip(f"missing required columns: {sorted(missing)}")

    checked = 0
    mismatches = []

    for index, row in frame.iterrows():
        home_score = _to_float(row.get("home_score"))
        away_score = _to_float(row.get("away_score"))
        home_win = _to_binary(row.get("home_win"))

        if home_score is None or away_score is None or home_win is None:
            continue

        if home_score == away_score:
            continue

        checked += 1
        expected = 1 if home_score > away_score else 0
        if home_win != expected:
            mismatches.append(
                {
                    "index": int(index),
                    "game_id": row.get("game_id"),
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_win": home_win,
                    "expected": expected,
                }
            )

    if checked == 0:
        pytest.skip("no finalized rows with scores and home_win")

    assert not mismatches, f"found reversed/mismatched home_win labels: {mismatches[:5]}"


def test_outcome_agrees_with_home_win_when_present() -> None:
    frame = _load_finalized()

    if "outcome" not in frame.columns or "home_win" not in frame.columns:
        pytest.skip("outcome and home_win columns are not both present")

    checked = 0
    mismatches = []

    for index, row in frame.iterrows():
        home_win = _to_binary(row.get("home_win"))
        outcome = _to_binary(row.get("outcome"))

        if home_win is None or outcome is None:
            continue

        checked += 1
        if home_win != outcome:
            mismatches.append(
                {
                    "index": int(index),
                    "game_id": row.get("game_id"),
                    "home_win": home_win,
                    "outcome": row.get("outcome"),
                }
            )

    if checked == 0:
        pytest.skip("no comparable outcome/home_win rows")

    assert not mismatches, f"outcome column disagrees with home_win: {mismatches[:5]}"
