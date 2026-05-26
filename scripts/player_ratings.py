"""Pitcher rating feature calculation with DataFrame and single-game support."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _numeric_value(payload: Any, names: tuple[str, ...]) -> float | None:
    for name in names:
        value = None
        if isinstance(payload, pd.Series):
            value = payload.get(name)
        elif isinstance(payload, dict):
            value = payload.get(name)
        if value is not None:
            parsed = pd.to_numeric(value, errors="coerce")
            if not pd.isna(parsed):
                return float(parsed)
    return None


def _single_game_rating(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    direct = _numeric_value(row, ("pitcher_rating_diff", "sp_rating_diff"))
    if direct is not None:
        return {"pitcher_rating_diff": float(direct)}

    home_rating = _numeric_value(
        row,
        ("home_pitcher_rating", "home_sp_rating", "home_stuff_plus", "home_stuff_plus_rating"),
    )
    away_rating = _numeric_value(
        row,
        ("away_pitcher_rating", "away_sp_rating", "away_stuff_plus", "away_stuff_plus_rating"),
    )
    if home_rating is not None and away_rating is not None:
        return {"pitcher_rating_diff": float(home_rating - away_rating)}

    home_era = _numeric_value(
        row,
        ("home_sp_era", "home_era", "home_pitcher_era", "home_pitcher_season_era"),
    )
    away_era = _numeric_value(
        row,
        ("away_sp_era", "away_era", "away_pitcher_era", "away_pitcher_season_era"),
    )
    if home_era is not None and away_era is not None:
        # Lower ERA is better; a one-run ERA advantage maps to +10 rating units.
        return {"pitcher_rating_diff": float(np.clip((away_era - home_era) * 10.0, -50.0, 50.0))}

    return {"pitcher_rating_diff": 0.0}


def calculate_pitcher_ratings(savant_df):
    """Calculate ratings from either one-game pitcher data or Statcast rows.

    prediction.py passes a single-game Series; the original historical feature
    workflow may still pass a DataFrame. Both paths are supported.
    """
    if isinstance(savant_df, (pd.Series, dict)):
        return _single_game_rating(savant_df)

    if savant_df is None or not isinstance(savant_df, pd.DataFrame) or savant_df.empty:
        return {}

    required_cols = [
        "release_speed",
        "pfx_x",
        "pfx_z",
        "barrel",
        "hard_hit",
        "release_spin_rate",
    ]
    if not all(column in savant_df.columns for column in required_cols):
        return {}

    df = savant_df.dropna(subset=required_cols).copy()
    if df.empty:
        return {}

    df["velocity_score"] = (df["release_speed"].astype(float) - 93.0) * 2.0 + 100.0
    movement = np.sqrt(df["pfx_x"].astype(float) ** 2 + df["pfx_z"].astype(float) ** 2)
    df["movement_score"] = (movement - 10.0) * 5.0 + 100.0
    df["swing_miss_score"] = (df["release_spin_rate"].astype(float) - 2300.0) * 0.05 + 100.0
    df["damage_score"] = (
        100.0 - df["barrel"].astype(float) * 50.0 - df["hard_hit"].astype(float) * 25.0
    )
    df["overall_rating"] = (
        df["velocity_score"]
        + df["movement_score"]
        + df["swing_miss_score"]
        + df["damage_score"]
    ) / 4.0
    df["overall_rating"] = df["overall_rating"].clip(0.0, 200.0)

    team_column = "home_team" if "home_team" in df.columns else ("team" if "team" in df.columns else None)
    if team_column is None:
        return {}

    team_ratings = df.groupby(team_column)["overall_rating"].mean()
    return {str(team): float(value) for team, value in team_ratings.items()}
