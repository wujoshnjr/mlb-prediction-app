"""Bullpen availability scoring with backward-compatible calling conventions."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _row_team_keys(row: pd.Series) -> list[str]:
    keys = []
    for column in ("team", "team_name", "club", "name", "team_id"):
        if column in row and _text(row.get(column)):
            keys.append(_text(row.get(column)))
    return keys


def _calculate_scores(bullpen_usage_df: pd.DataFrame) -> dict[str, float]:
    if bullpen_usage_df is None or bullpen_usage_df.empty:
        return {}

    scores: dict[str, float] = {}
    for _, row in bullpen_usage_df.iterrows():
        pitches = pd.to_numeric(row.get("bullpen_pitches", row.get("pitches", 0)), errors="coerce")
        innings = pd.to_numeric(row.get("bullpen_innings", row.get("innings", 0)), errors="coerce")
        back_to_back = pd.to_numeric(row.get("back_to_back", 0), errors="coerce")

        pitches = 0.0 if pd.isna(pitches) else float(pitches)
        innings = 0.0 if pd.isna(innings) else float(innings)
        back_to_back = 0.0 if pd.isna(back_to_back) else float(back_to_back)

        # Higher means fresher bullpen. This preserves the original intent while
        # making the score bounded and stable.
        score = max(0.0, min(100.0, 100.0 - pitches * 0.6 - innings * 3.0 - back_to_back * 15.0))
        for key in _row_team_keys(row):
            scores[key] = score

    return scores


def _lookup(scores: dict[str, float], team: Any) -> float:
    key = _text(team)
    if key in scores:
        return scores[key]

    # Support full-team-name versus short-name joins, e.g. Cleveland Guardians
    # versus Guardians.
    for candidate, score in scores.items():
        if candidate.endswith(key) or key.endswith(candidate):
            return score
    return 50.0


def calculate_bullpen_availability(
    bullpen_usage_df: pd.DataFrame,
    home_team: Any = None,
    away_team: Any = None,
):
    """Return team scores or a home-away comparison.

    Original callers may pass only a DataFrame and receive a score dictionary.
    prediction.py passes (DataFrame, home_team, away_team), so that path now
    returns a feature dictionary including bullpen_availability_diff.
    """
    scores = _calculate_scores(bullpen_usage_df)

    if home_team is None and away_team is None:
        return scores

    home_score = _lookup(scores, home_team)
    away_score = _lookup(scores, away_team)
    return {
        "home_availability": home_score,
        "away_availability": away_score,
        "home_score": home_score,
        "away_score": away_score,
        "bullpen_availability_diff": float(home_score - away_score),
    }
