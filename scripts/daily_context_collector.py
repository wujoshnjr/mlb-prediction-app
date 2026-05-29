from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.bullpen_context_client import fetch_bullpen_context
from scripts.daily_game_context import (
    append_context_snapshots,
    parse_utc_datetime,
)
from scripts.lineup_client import fetch_confirmed_lineups
from scripts.pitcher_client import fetch_probable_pitchers


def _utc_now_iso() -> str:
    """Return the current UTC timestamp ending in Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_missing(value: Any) -> bool:
    """Return True when a scalar value should be treated as unavailable."""
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    try:
        missing = pd.isna(value)
        if isinstance(missing, bool):
            return missing
    except (TypeError, ValueError):
        pass

    return False


def _optional_value(value: Any) -> Any:
    """Convert pandas missing values to None while preserving valid values."""
    if _is_missing(value):
        return None
    return value


def _optional_bool(value: Any) -> Optional[bool]:
    """Normalize optional boolean values from dataframe rows."""
    if _is_missing(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return None

    try:
        if value == 1:
            return True
        if value == 0:
            return False
    except Exception:
        return None

    return None


def _string_key(value: Any) -> str:
    """Convert an identifier into a stable string lookup key."""
    if _is_missing(value):
        return ""
    return str(value)


def _record_by_key(
    frame: pd.DataFrame,
    key_column: str,
) -> Dict[str, Dict[str, Any]]:
    """Build a lookup dictionary from a dataframe using one identifier column."""
    lookup: Dict[str, Dict[str, Any]] = {}

    if frame.empty or key_column not in frame.columns:
        return lookup

    for _, row in frame.iterrows():
        key = _string_key(row.get(key_column))
        if not key:
            continue
        lookup[key] = row.to_dict()

    return lookup


def _build_data_sources(
    *,
    pitcher_row: Dict[str, Any],
    lineup_row: Optional[Dict[str, Any]],
    home_bullpen_row: Optional[Dict[str, Any]],
    away_bullpen_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Describe which source payloads were available for one context row."""
    sources: Dict[str, Any] = {
        "pitcher": {
            "available": True,
            "source": pitcher_row.get("pitcher_source"),
            "fetched_at": pitcher_row.get("pitcher_fetched_at"),
        },
        "lineup": {
            "available": lineup_row is not None,
            "source": (
                lineup_row.get("lineup_source")
                if lineup_row is not None
                else None
            ),
            "fetched_at": (
                lineup_row.get("lineup_fetched_at")
                if lineup_row is not None
                else None
            ),
        },
        "bullpen": {
            "home_available": home_bullpen_row is not None,
            "away_available": away_bullpen_row is not None,
            "home_source": (
                home_bullpen_row.get("bullpen_source")
                if home_bullpen_row is not None
                else None
            ),
            "away_source": (
                away_bullpen_row.get("bullpen_source")
                if away_bullpen_row is not None
                else None
            ),
            "home_captured_at": (
                home_bullpen_row.get("captured_at")
                if home_bullpen_row is not None
                else None
            ),
            "away_captured_at": (
                away_bullpen_row.get("captured_at")
                if away_bullpen_row is not None
                else None
            ),
        },
    }

    return sources


def _build_context_for_game(
    *,
    pitcher_row: Dict[str, Any],
    lineup_row: Optional[Dict[str, Any]],
    home_bullpen_row: Optional[Dict[str, Any]],
    away_bullpen_row: Optional[Dict[str, Any]],
    captured_at: str,
) -> Optional[Dict[str, Any]]:
    """Build one daily context input row for an upcoming scheduled game."""
    game_id = pitcher_row.get("game_id")
    start_time = pitcher_row.get("start_time")

    if _is_missing(game_id) or _is_missing(start_time):
        return None

    start_dt = parse_utc_datetime(start_time)
    captured_dt = parse_utc_datetime(captured_at)

    if start_dt is None or captured_dt is None:
        return None

    if captured_dt >= start_dt:
        return None

    lineup_row = lineup_row or {}
    home_bullpen_row = home_bullpen_row or {}
    away_bullpen_row = away_bullpen_row or {}

    context: Dict[str, Any] = {
        "game_id": game_id,
        "game_date": pitcher_row.get("game_date", ""),
        "start_time": start_time,
        "captured_at": captured_at,
        "home_team": pitcher_row.get("home_team", ""),
        "away_team": pitcher_row.get("away_team", ""),

        "home_probable_pitcher_id": _optional_value(
            pitcher_row.get("home_probable_pitcher_id")
        ),
        "away_probable_pitcher_id": _optional_value(
            pitcher_row.get("away_probable_pitcher_id")
        ),
        "home_probable_pitcher_name": pitcher_row.get(
            "home_probable_pitcher_name",
            "",
        ),
        "away_probable_pitcher_name": pitcher_row.get(
            "away_probable_pitcher_name",
            "",
        ),

        # Probable pitchers are not official pregame confirmed starters.
        "home_starting_pitcher_confirmed": None,
        "away_starting_pitcher_confirmed": None,

        "home_sp_season_era": _optional_value(
            pitcher_row.get("home_era")
        ),
        "away_sp_season_era": _optional_value(
            pitcher_row.get("away_era")
        ),
        "home_sp_season_fip": _optional_value(
            pitcher_row.get("home_fip")
        ),
        "away_sp_season_fip": _optional_value(
            pitcher_row.get("away_fip")
        ),

        "home_lineup_confirmed": _optional_bool(
            lineup_row.get("home_lineup_confirmed")
        ),
        "away_lineup_confirmed": _optional_bool(
            lineup_row.get("away_lineup_confirmed")
        ),
        "home_lineup_player_ids_json": lineup_row.get(
            "home_lineup_player_ids_json",
            [],
        ),
        "away_lineup_player_ids_json": lineup_row.get(
            "away_lineup_player_ids_json",
            [],
        ),

        "home_bullpen_data_available": _optional_bool(
            home_bullpen_row.get("bullpen_data_available")
        ),
        "away_bullpen_data_available": _optional_bool(
            away_bullpen_row.get("bullpen_data_available")
        ),
        "home_bullpen_pitches_last_1d": _optional_value(
            home_bullpen_row.get("reliever_pitches_last_1d")
        ),
        "away_bullpen_pitches_last_1d": _optional_value(
            away_bullpen_row.get("reliever_pitches_last_1d")
        ),
        "home_bullpen_pitches_last_3d": _optional_value(
            home_bullpen_row.get("reliever_pitches_last_3d")
        ),
        "away_bullpen_pitches_last_3d": _optional_value(
            away_bullpen_row.get("reliever_pitches_last_3d")
        ),
        "home_bullpen_fatigue_score": _optional_value(
            home_bullpen_row.get("bullpen_fatigue_score")
        ),
        "away_bullpen_fatigue_score": _optional_value(
            away_bullpen_row.get("bullpen_fatigue_score")
        ),
        "home_extra_innings_previous_game": _optional_bool(
            home_bullpen_row.get("extra_innings_previous_game")
        ),
        "away_extra_innings_previous_game": _optional_bool(
            away_bullpen_row.get("extra_innings_previous_game")
        ),

        # This remains unknown until an audited closer availability method exists.
        "home_closer_available": _optional_bool(
            home_bullpen_row.get("closer_available_estimate")
        ),
        "away_closer_available": _optional_bool(
            away_bullpen_row.get("closer_available_estimate")
        ),

        "context_data_sources_json": _build_data_sources(
            pitcher_row=pitcher_row,
            lineup_row=lineup_row if lineup_row else None,
            home_bullpen_row=(
                home_bullpen_row if home_bullpen_row else None
            ),
            away_bullpen_row=(
                away_bullpen_row if away_bullpen_row else None
            ),
        ),
    }

    return context


def collect_daily_context(
    date_str: Optional[str] = None,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Collect and append pregame daily context snapshots without betting use."""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    local_errors: List[str] = []
    error_sink = errors if errors is not None else local_errors

    captured_at = _utc_now_iso()

    pitcher_frame = fetch_probable_pitchers(
        date_str=date_str,
        errors=error_sink,
    )

    if pitcher_frame.empty:
        return {
            "date": date_str,
            "captured_at": captured_at,
            "games_received": 0,
            "context_rows_built": 0,
            "append_summary": {
                "received": 0,
                "inserted": 0,
                "duplicates": 0,
                "stored_rows": 0,
                "errors": [],
            },
            "errors": list(error_sink),
        }

    lineup_frame = fetch_confirmed_lineups(
        date_str=date_str,
        errors=error_sink,
    )

    bullpen_frame = fetch_bullpen_context(
        as_of_date=date_str,
        errors=error_sink,
    )

    lineup_by_game = _record_by_key(lineup_frame, "game_id")
    bullpen_by_team = _record_by_key(bullpen_frame, "team_id")

    contexts: List[Dict[str, Any]] = []

    for _, pitcher_series in pitcher_frame.iterrows():
        pitcher_row = pitcher_series.to_dict()

        game_key = _string_key(pitcher_row.get("game_id"))
        home_team_key = _string_key(pitcher_row.get("home_team_id"))
        away_team_key = _string_key(pitcher_row.get("away_team_id"))

        context = _build_context_for_game(
            pitcher_row=pitcher_row,
            lineup_row=lineup_by_game.get(game_key),
            home_bullpen_row=bullpen_by_team.get(home_team_key),
            away_bullpen_row=bullpen_by_team.get(away_team_key),
            captured_at=captured_at,
        )

        if context is not None:
            contexts.append(context)

    append_summary = append_context_snapshots(contexts)

    return {
        "date": date_str,
        "captured_at": captured_at,
        "games_received": int(len(pitcher_frame)),
        "context_rows_built": int(len(contexts)),
        "append_summary": append_summary,
        "errors": list(error_sink),
    }


def main() -> None:
    """Run context collection for today's UTC games and print a JSON summary."""
    summary = collect_daily_context()
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
