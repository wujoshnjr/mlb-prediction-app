from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.bullpen_context_client import fetch_bullpen_context
from scripts.daily_game_context import (
    append_context_snapshots,
    parse_utc_datetime,
)
from scripts.lineup_client import fetch_confirmed_lineups
from scripts.mlb_game_feed_client import fetch_mlb_game_feed_contexts
from scripts.pitcher_client import fetch_probable_pitchers

DAILY_CONTEXT_COLLECTION_REPORT = Path("report/daily_context_collection_report.json")

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
    game_feed_row: Optional[Dict[str, Any]],
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
        "game_feed": {
            "available": game_feed_row is not None
            and bool(game_feed_row.get("game_feed_available")),
            "source": "mlb_statsapi_feed_live",
            "fetched_at": (
                game_feed_row.get("game_feed_captured_at")
                if game_feed_row is not None
                else None
            ),
            "error": (
                game_feed_row.get("game_feed_error")
                if game_feed_row is not None
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
    game_feed_row: Optional[Dict[str, Any]],
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
    game_feed_row = game_feed_row or {}
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

        # Game feed can confirm starters after official lineup / game state appears.
        "home_starting_pitcher_id": _optional_value(
            game_feed_row.get("home_starting_pitcher_id")
        ),
        "away_starting_pitcher_id": _optional_value(
            game_feed_row.get("away_starting_pitcher_id")
        ),
        "home_starting_pitcher_name": game_feed_row.get(
            "home_starting_pitcher_name",
            "",
        ),
        "away_starting_pitcher_name": game_feed_row.get(
            "away_starting_pitcher_name",
            "",
        ),
        "home_starting_pitcher_confirmed": _optional_bool(
            game_feed_row.get("home_starting_pitcher_confirmed")
        ),
        "away_starting_pitcher_confirmed": _optional_bool(
            game_feed_row.get("away_starting_pitcher_confirmed")
        ),
        
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

        "home_lineup_confirmed": (
            _optional_bool(lineup_row.get("home_lineup_confirmed"))
            if not _is_missing(lineup_row.get("home_lineup_confirmed"))
            else _optional_bool(game_feed_row.get("home_lineup_confirmed"))
        ),
        "away_lineup_confirmed": (
            _optional_bool(lineup_row.get("away_lineup_confirmed"))
            if not _is_missing(lineup_row.get("away_lineup_confirmed"))
            else _optional_bool(game_feed_row.get("away_lineup_confirmed"))
        ),
        "home_lineup_player_ids_json": (
            lineup_row.get("home_lineup_player_ids_json")
            if not _is_missing(lineup_row.get("home_lineup_player_ids_json"))
            else game_feed_row.get("home_batting_order_ids", "")
        ),
        "away_lineup_player_ids_json": (
            lineup_row.get("away_lineup_player_ids_json")
            if not _is_missing(lineup_row.get("away_lineup_player_ids_json"))
            else game_feed_row.get("away_batting_order_ids", "")
        ),
        "home_lineup_player_count": _optional_value(
            game_feed_row.get("home_lineup_player_count")
        ),
        "away_lineup_player_count": _optional_value(
            game_feed_row.get("away_lineup_player_count")
        ),
        "home_top3_player_ids": game_feed_row.get("home_top3_player_ids", ""),
        "away_top3_player_ids": game_feed_row.get("away_top3_player_ids", ""),
        "home_catcher_id": _optional_value(game_feed_row.get("home_catcher_id")),
        "away_catcher_id": _optional_value(game_feed_row.get("away_catcher_id")),
        "home_catcher_name": game_feed_row.get("home_catcher_name", ""),
        "away_catcher_name": game_feed_row.get("away_catcher_name", ""),

        "venue_id": _optional_value(game_feed_row.get("venue_id")),
        "venue_name": game_feed_row.get("venue_name", ""),
        "weather_temp": _optional_value(game_feed_row.get("weather_temp")),
        "weather_condition": game_feed_row.get("weather_condition", ""),
        "wind_speed": _optional_value(game_feed_row.get("wind_speed")),
        "wind_direction": game_feed_row.get("wind_direction", ""),
        "umpire_home_plate_id": _optional_value(
            game_feed_row.get("umpire_home_plate_id")
        ),
        "umpire_home_plate_name": game_feed_row.get(
            "umpire_home_plate_name",
            "",
        ),
        "game_feed_available": _optional_bool(
            game_feed_row.get("game_feed_available")
        ),
        "game_feed_error": game_feed_row.get("game_feed_error", ""),
        "game_feed_captured_at": game_feed_row.get("game_feed_captured_at", ""),
        
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
            game_feed_row=game_feed_row if game_feed_row else None,
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

    game_ids = [
        _string_key(value)
        for value in pitcher_frame.get("game_id", [])
        if _string_key(value)
    ]

    game_feed_frame = fetch_mlb_game_feed_contexts(
        game_ids=game_ids,
        errors=error_sink,
        timeout=15,
        sleep_seconds=0.1,
    )

    bullpen_frame = fetch_bullpen_context(
        as_of_date=date_str,
        errors=error_sink,
    )

    lineup_by_game = _record_by_key(lineup_frame, "game_id")
    game_feed_by_game = _record_by_key(game_feed_frame, "game_id")
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
            game_feed_row=game_feed_by_game.get(game_key),
            home_bullpen_row=bullpen_by_team.get(home_team_key),
            away_bullpen_row=bullpen_by_team.get(away_team_key),
            captured_at=captured_at,
        )
        
        if context is not None:
            contexts.append(context)

    append_summary = append_context_snapshots(contexts)

    sample_context_keys = sorted(contexts[0].keys()) if contexts else []
    sample_context = contexts[0] if contexts else {}

    summary = {
        "date": date_str,
        "captured_at": captured_at,
        "games_received": int(len(pitcher_frame)),
        "game_feed_rows_received": int(len(game_feed_frame)),
        "context_rows_built": int(len(contexts)),
        "sample_context_keys": sample_context_keys,
        "sample_game_feed_available": sample_context.get("game_feed_available"),
        "sample_game_feed_error": sample_context.get("game_feed_error"),
        "sample_home_lineup_player_count": sample_context.get(
            "home_lineup_player_count"
        ),
        "sample_home_top3_player_ids": sample_context.get("home_top3_player_ids"),
        "append_summary": append_summary,
        "errors": list(error_sink),
    }

    try:
        DAILY_CONTEXT_COLLECTION_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with DAILY_CONTEXT_COLLECTION_REPORT.open("w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        error_sink.append(f"Failed to write daily context report: {exc}")
        summary["errors"] = list(error_sink)

    return summary


def main() -> None:
    """Run context collection for today's UTC games and print a JSON summary."""
    summary = collect_daily_context()
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
