from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PIPELINE_VERSION = "baseline_v2_clean"
CONTEXT_SCHEMA_VERSION = "daily_context_v2"
CONTEXT_STORE_FILE = Path("data/daily_game_context.csv")

COLUMNS = [
    "context_snapshot_id",
    "pipeline_version",
    "context_schema_version",
    "game_id",
    "game_date",
    "start_time",
    "captured_at",
    "minutes_to_start",
    "is_pregame",
    "home_team",
    "away_team",
    "home_probable_pitcher_id",
    "home_probable_pitcher_name",
    "away_probable_pitcher_id",
    "away_probable_pitcher_name",
    "home_starting_pitcher_id",
    "away_starting_pitcher_id",
    "home_starting_pitcher_name",
    "away_starting_pitcher_name",
    "home_starting_pitcher_confirmed",
    "away_starting_pitcher_confirmed",
    "starting_pitchers_ready",
    "home_starter_status",
    "away_starter_status",
    "home_starter_confidence",
    "away_starter_confidence",
    "home_starter_confidence_score",
    "away_starter_confidence_score",
    "home_starter_reason",
    "away_starter_reason",
    "home_sp_season_era",
    "away_sp_season_era",
    "home_sp_season_fip",
    "away_sp_season_fip",
    "home_sp_season_xwoba_allowed",
    "away_sp_season_xwoba_allowed",
    "home_sp_last3_xwoba_allowed",
    "away_sp_last3_xwoba_allowed",
    "home_sp_fastball_velocity_change",
    "away_sp_fastball_velocity_change",
    "home_sp_k_minus_bb_pct",
    "away_sp_k_minus_bb_pct",
    "home_sp_days_rest",
    "away_sp_days_rest",
    "home_lineup_confirmed",
    "away_lineup_confirmed",
    "lineups_ready",
    "home_lineup_player_ids_json",
    "away_lineup_player_ids_json",
    "home_lineup_player_count",
    "away_lineup_player_count",
    "home_top3_player_ids",
    "away_top3_player_ids",
    "home_catcher_id",
    "away_catcher_id",
    "home_catcher_name",
    "away_catcher_name",
    "home_confirmed_lineup_xwoba",
    "away_confirmed_lineup_xwoba",
    "home_lineup_ops_vs_hand",
    "away_lineup_ops_vs_hand",
    "home_lineup_strength_change_vs_expected",
    "away_lineup_strength_change_vs_expected",
    "home_top_bat_missing_count",
    "away_top_bat_missing_count",
    "home_bullpen_data_available",
    "away_bullpen_data_available",
    "bullpens_ready",
    "home_bullpen_pitches_last_1d",
    "away_bullpen_pitches_last_1d",
    "home_bullpen_pitches_last_3d",
    "away_bullpen_pitches_last_3d",
    "home_high_leverage_pitches_last_3d",
    "away_high_leverage_pitches_last_3d",
    "home_closer_available",
    "away_closer_available",
    "home_high_leverage_available_count",
    "away_high_leverage_available_count",
    "home_bullpen_fatigue_score",
    "away_bullpen_fatigue_score",
    "home_extra_innings_previous_game",
    "away_extra_innings_previous_game",
    "weather_data_available",
    "park_factor_available",
    "venue_id",
    "venue_name",
    "weather_temp",
    "weather_condition",
    "wind_speed",
    "wind_direction",
    "umpire_home_plate_id",
    "umpire_home_plate_name",
    "game_feed_available",
    "game_feed_error",
    "game_feed_captured_at",
    "context_data_sources_json",
    "missing_critical_fields_json",
    "data_completeness_score",
    "context_ready_for_betting",
    "context_not_ready_reason",
]

SOURCE_BOOL_FIELDS = [
    "home_starting_pitcher_confirmed",
    "away_starting_pitcher_confirmed",
    "home_starter_confidence",
    "away_starter_confidence",
    "home_lineup_confirmed",
    "away_lineup_confirmed",
    "home_bullpen_data_available",
    "away_bullpen_data_available",
    "home_closer_available",
    "away_closer_available",
    "home_extra_innings_previous_game",
    "away_extra_innings_previous_game",
    "weather_data_available",
    "park_factor_available",
    "game_feed_available",
]

DERIVED_BOOL_FIELDS = [
    "is_pregame",
    "starting_pitchers_ready",
    "lineups_ready",
    "bullpens_ready",
    "context_ready_for_betting",
]

BOOL_FIELDS = SOURCE_BOOL_FIELDS + DERIVED_BOOL_FIELDS


def parse_utc_datetime(value: Any) -> Optional[datetime]:
    """Convert an ISO timestamp or datetime into a UTC-aware datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if not isinstance(value, str):
        return None

    try:
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> Optional[bool]:
    """Interpret a value as bool without converting unknown values to True."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return None

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None

    return None


def _is_missing(value: Any) -> bool:
    """Return True when a scalar value should be treated as missing."""
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


def _clean_json_value(value: Any) -> Any:
    """Convert values into JSON-safe values without persisting NaN."""
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): _clean_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_clean_json_value(item) for item in value]

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    if isinstance(value, str):
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def _safe_json_dumps(value: Any) -> str:
    """Serialize an optional JSON field while preventing double encoding."""
    if _is_missing(value):
        return ""

    normalized: Any = value

    if isinstance(value, str):
        try:
            normalized = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            normalized = value

    normalized = _clean_json_value(normalized)

    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def ensure_context_store() -> None:
    """Create or safely migrate the canonical context history CSV."""
    if not CONTEXT_STORE_FILE.exists():
        CONTEXT_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=COLUMNS).to_csv(
            CONTEXT_STORE_FILE,
            index=False,
            encoding="utf-8",
        )
        return

    try:
        existing = pd.read_csv(CONTEXT_STORE_FILE)
    except Exception as exc:
        raise RuntimeError(
            "Cannot read existing context store; refusing to overwrite "
            f"{CONTEXT_STORE_FILE}: {exc}"
        ) from exc

    existing_columns = list(existing.columns)
    missing_columns = [
        column for column in COLUMNS if column not in existing_columns
    ]

    if not missing_columns:
        return

    for column in missing_columns:
        existing[column] = None

    extra_columns = [
        column for column in existing_columns if column not in COLUMNS
    ]

    existing = existing[COLUMNS + extra_columns]
    existing.to_csv(
        CONTEXT_STORE_FILE,
        index=False,
        encoding="utf-8",
    )


def _load_context_store() -> pd.DataFrame:
    """Read context history and normalize identifiers and boolean fields."""
    if not CONTEXT_STORE_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)

    ensure_context_store()

    try:
        frame = pd.read_csv(CONTEXT_STORE_FILE)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read context store {CONTEXT_STORE_FILE}: {exc}"
        ) from exc

    if frame.empty:
        return frame

    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str)

    for column in BOOL_FIELDS:
        if column in frame.columns:
            frame[column] = frame[column].apply(_safe_bool)

    return frame


def calculate_data_completeness(
    normalized_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate whether critical pregame context is complete."""
    start_time = parse_utc_datetime(normalized_context.get("start_time"))
    captured_at = parse_utc_datetime(normalized_context.get("captured_at"))

    is_valid_pregame_time = (
        start_time is not None
        and captured_at is not None
        and captured_at < start_time
    )

    critical_checks = [
        ("game_id", bool(normalized_context.get("game_id"))),
        ("pregame_timing", is_valid_pregame_time),
        (
            "home_starting_pitcher_confirmed",
            normalized_context.get("home_starting_pitcher_confirmed")
            is True,
        ),
        (
            "away_starting_pitcher_confirmed",
            normalized_context.get("away_starting_pitcher_confirmed")
            is True,
        ),
        (
            "home_lineup_confirmed",
            normalized_context.get("home_lineup_confirmed") is True,
        ),
        (
            "away_lineup_confirmed",
            normalized_context.get("away_lineup_confirmed") is True,
        ),
        (
            "home_bullpen_data_available",
            normalized_context.get("home_bullpen_data_available") is True,
        ),
        (
            "away_bullpen_data_available",
            normalized_context.get("away_bullpen_data_available") is True,
        ),
        (
            "home_closer_available_known",
            normalized_context.get("home_closer_available") is not None,
        ),
        (
            "away_closer_available_known",
            normalized_context.get("away_closer_available") is not None,
        ),
    ]

    missing_critical_fields = [
        name for name, passed in critical_checks if not passed
    ]
    total_critical = len(critical_checks)
    passed_critical = total_critical - len(missing_critical_fields)

    data_completeness_score = round(
        passed_critical / total_critical,
        4,
    )
    context_ready_for_betting = not missing_critical_fields

    if context_ready_for_betting:
        context_not_ready_reason = ""
    else:
        context_not_ready_reason = (
            "Missing critical fields: "
            + ", ".join(missing_critical_fields)
        )

    return {
        "missing_critical_fields": missing_critical_fields,
        "data_completeness_score": data_completeness_score,
        "context_ready_for_betting": context_ready_for_betting,
        "context_not_ready_reason": context_not_ready_reason,
    }


def build_context_snapshot_row(context: Dict[str, Any]) -> Dict[str, Any]:
    """Build one validated pregame context row from caller-provided data."""
    game_id = context.get("game_id")
    if _is_missing(game_id):
        raise ValueError("game_id is missing or empty")

    start_time = parse_utc_datetime(context.get("start_time"))
    captured_at = parse_utc_datetime(context.get("captured_at"))

    if start_time is None:
        raise ValueError("start_time is missing or unparseable")

    if captured_at is None:
        raise ValueError("captured_at is missing or unparseable")

    if captured_at >= start_time:
        raise ValueError(
            "Snapshot is not pregame: captured_at >= start_time"
        )

    minutes_to_start = (
        start_time - captured_at
    ).total_seconds() / 60.0

    source_booleans = {
        field: _safe_bool(context.get(field))
        for field in SOURCE_BOOL_FIELDS
    }

    starting_pitchers_ready = (
        source_booleans["home_starting_pitcher_confirmed"] is True
        and source_booleans["away_starting_pitcher_confirmed"] is True
    )

    lineups_ready = (
        source_booleans["home_lineup_confirmed"] is True
        and source_booleans["away_lineup_confirmed"] is True
    )

    bullpens_ready = (
        source_booleans["home_bullpen_data_available"] is True
        and source_booleans["away_bullpen_data_available"] is True
        and source_booleans["home_closer_available"] is not None
        and source_booleans["away_closer_available"] is not None
    )

    normalized_context = {
        "game_id": str(game_id),
        "start_time": start_time.isoformat(),
        "captured_at": captured_at.isoformat(),
        **source_booleans,
    }

    completeness = calculate_data_completeness(normalized_context)

    snapshot_key = (
        f"{PIPELINE_VERSION}|{CONTEXT_SCHEMA_VERSION}|"
        f"{game_id}|{captured_at.isoformat()}"
    )
    context_snapshot_id = hashlib.sha256(
        snapshot_key.encode("ascii", errors="ignore")
    ).hexdigest()

    row: Dict[str, Any] = {
        "context_snapshot_id": context_snapshot_id,
        "pipeline_version": PIPELINE_VERSION,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "game_id": str(game_id),
        "game_date": context.get("game_date", ""),
        "start_time": start_time.isoformat(),
        "captured_at": captured_at.isoformat(),
        "minutes_to_start": minutes_to_start,
        "is_pregame": True,
        "home_team": context.get("home_team", ""),
        "away_team": context.get("away_team", ""),
        "home_probable_pitcher_id": context.get(
            "home_probable_pitcher_id",
            "",
        ),
        "home_probable_pitcher_name": context.get(
            "home_probable_pitcher_name",
            "",
        ),
        "away_probable_pitcher_id": context.get(
            "away_probable_pitcher_id",
            "",
        ),
        "away_probable_pitcher_name": context.get(
            "away_probable_pitcher_name",
            "",
        ),
        "home_starting_pitcher_id": context.get("home_starting_pitcher_id"),
        "away_starting_pitcher_id": context.get("away_starting_pitcher_id"),
        "home_starting_pitcher_name": context.get(
            "home_starting_pitcher_name",
            "",
        ),
        "away_starting_pitcher_name": context.get(
            "away_starting_pitcher_name",
            "",
        ),
        "home_starting_pitcher_confirmed": source_booleans[
            "home_starting_pitcher_confirmed"
        ],
        "away_starting_pitcher_confirmed": source_booleans[
            "away_starting_pitcher_confirmed"
        ],
        "starting_pitchers_ready": starting_pitchers_ready,
        "home_starter_status": context.get("home_starter_status", ""),
        "away_starter_status": context.get("away_starter_status", ""),
        "home_starter_confidence": source_booleans[
            "home_starter_confidence"
        ],
        "away_starter_confidence": source_booleans[
            "away_starter_confidence"
        ],
        "home_starter_confidence_score": context.get(
            "home_starter_confidence_score"
        ),
        "away_starter_confidence_score": context.get(
            "away_starter_confidence_score"
        ),
        "home_starter_reason": context.get("home_starter_reason", ""),
        "away_starter_reason": context.get("away_starter_reason", ""),
        "home_sp_season_era": context.get("home_sp_season_era"),
        "away_sp_season_era": context.get("away_sp_season_era"),
        "home_sp_season_fip": context.get("home_sp_season_fip"),
        "away_sp_season_fip": context.get("away_sp_season_fip"),
        "home_sp_season_xwoba_allowed": context.get(
            "home_sp_season_xwoba_allowed"
        ),
        "away_sp_season_xwoba_allowed": context.get(
            "away_sp_season_xwoba_allowed"
        ),
        "home_sp_last3_xwoba_allowed": context.get(
            "home_sp_last3_xwoba_allowed"
        ),
        "away_sp_last3_xwoba_allowed": context.get(
            "away_sp_last3_xwoba_allowed"
        ),
        "home_sp_fastball_velocity_change": context.get(
            "home_sp_fastball_velocity_change"
        ),
        "away_sp_fastball_velocity_change": context.get(
            "away_sp_fastball_velocity_change"
        ),
        "home_sp_k_minus_bb_pct": context.get(
            "home_sp_k_minus_bb_pct"
        ),
        "away_sp_k_minus_bb_pct": context.get(
            "away_sp_k_minus_bb_pct"
        ),
        "home_sp_days_rest": context.get("home_sp_days_rest"),
        "away_sp_days_rest": context.get("away_sp_days_rest"),
        "home_lineup_confirmed": source_booleans[
            "home_lineup_confirmed"
        ],
        "away_lineup_confirmed": source_booleans[
            "away_lineup_confirmed"
        ],
        "lineups_ready": lineups_ready,
        "home_lineup_player_ids_json": _safe_json_dumps(
            context.get("home_lineup_player_ids_json")
        ),
        "away_lineup_player_ids_json": _safe_json_dumps(
            context.get("away_lineup_player_ids_json")
        ),
        "home_lineup_player_count": context.get("home_lineup_player_count"),
        "away_lineup_player_count": context.get("away_lineup_player_count"),
        "home_top3_player_ids": context.get("home_top3_player_ids", ""),
        "away_top3_player_ids": context.get("away_top3_player_ids", ""),
        "home_catcher_id": context.get("home_catcher_id"),
        "away_catcher_id": context.get("away_catcher_id"),
        "home_catcher_name": context.get("home_catcher_name", ""),
        "away_catcher_name": context.get("away_catcher_name", ""),
        "home_confirmed_lineup_xwoba": context.get(
            "home_confirmed_lineup_xwoba"
        ),
        "away_confirmed_lineup_xwoba": context.get(
            "away_confirmed_lineup_xwoba"
        ),
        "home_lineup_ops_vs_hand": context.get(
            "home_lineup_ops_vs_hand"
        ),
        "away_lineup_ops_vs_hand": context.get(
            "away_lineup_ops_vs_hand"
        ),
        "home_lineup_strength_change_vs_expected": context.get(
            "home_lineup_strength_change_vs_expected"
        ),
        "away_lineup_strength_change_vs_expected": context.get(
            "away_lineup_strength_change_vs_expected"
        ),
        "home_top_bat_missing_count": context.get(
            "home_top_bat_missing_count"
        ),
        "away_top_bat_missing_count": context.get(
            "away_top_bat_missing_count"
        ),
        "home_bullpen_data_available": source_booleans[
            "home_bullpen_data_available"
        ],
        "away_bullpen_data_available": source_booleans[
            "away_bullpen_data_available"
        ],
        "bullpens_ready": bullpens_ready,
        "home_bullpen_pitches_last_1d": context.get(
            "home_bullpen_pitches_last_1d"
        ),
        "away_bullpen_pitches_last_1d": context.get(
            "away_bullpen_pitches_last_1d"
        ),
        "home_bullpen_pitches_last_3d": context.get(
            "home_bullpen_pitches_last_3d"
        ),
        "away_bullpen_pitches_last_3d": context.get(
            "away_bullpen_pitches_last_3d"
        ),
        "home_high_leverage_pitches_last_3d": context.get(
            "home_high_leverage_pitches_last_3d"
        ),
        "away_high_leverage_pitches_last_3d": context.get(
            "away_high_leverage_pitches_last_3d"
        ),
        "home_closer_available": source_booleans[
            "home_closer_available"
        ],
        "away_closer_available": source_booleans[
            "away_closer_available"
        ],
        "home_high_leverage_available_count": context.get(
            "home_high_leverage_available_count"
        ),
        "away_high_leverage_available_count": context.get(
            "away_high_leverage_available_count"
        ),
        "home_bullpen_fatigue_score": context.get(
            "home_bullpen_fatigue_score"
        ),
        "away_bullpen_fatigue_score": context.get(
            "away_bullpen_fatigue_score"
        ),
        "home_extra_innings_previous_game": source_booleans[
            "home_extra_innings_previous_game"
        ],
        "away_extra_innings_previous_game": source_booleans[
            "away_extra_innings_previous_game"
        ],
        "weather_data_available": source_booleans[
            "weather_data_available"
        ],
        "park_factor_available": source_booleans[
            "park_factor_available"
        ],
        "venue_id": context.get("venue_id"),
        "venue_name": context.get("venue_name", ""),
        "weather_temp": context.get("weather_temp"),
        "weather_condition": context.get("weather_condition", ""),
        "wind_speed": context.get("wind_speed"),
        "wind_direction": context.get("wind_direction", ""),
        "umpire_home_plate_id": context.get("umpire_home_plate_id"),
        "umpire_home_plate_name": context.get(
            "umpire_home_plate_name",
            "",
        ),
        "game_feed_available": source_booleans["game_feed_available"],
        "game_feed_error": context.get("game_feed_error", ""),
        "game_feed_captured_at": context.get("game_feed_captured_at", ""),
        "context_data_sources_json": _safe_json_dumps(
            context.get("context_data_sources_json")
        ),
        "missing_critical_fields_json": _safe_json_dumps(
            completeness["missing_critical_fields"]
        ),
        "data_completeness_score": completeness[
            "data_completeness_score"
        ],
        "context_ready_for_betting": completeness[
            "context_ready_for_betting"
        ],
        "context_not_ready_reason": completeness[
            "context_not_ready_reason"
        ],
    }

    return {column: row.get(column) for column in COLUMNS}


def append_context_snapshots(
    contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate and append multiple daily context snapshots."""
    summary: Dict[str, Any] = {
        "received": len(contexts),
        "inserted": 0,
        "duplicates": 0,
        "stored_rows": 0,
        "errors": [],
    }

    valid_rows: List[Dict[str, Any]] = []

    for index, context in enumerate(contexts):
        try:
            valid_rows.append(build_context_snapshot_row(context))
        except ValueError as exc:
            summary["errors"].append(
                f"Context index {index} rejected: {exc}"
            )
        except Exception as exc:
            summary["errors"].append(
                f"Context index {index} unexpected error: {exc}"
            )

    try:
        ensure_context_store()
        existing = _load_context_store()
    except Exception as exc:
        summary["errors"].append(
            f"Failed to prepare context store: {exc}"
        )
        return summary

    summary["stored_rows"] = int(len(existing))

    if not valid_rows:
        return summary

    existing_columns = list(existing.columns)
    extra_columns = [
        column for column in existing_columns if column not in COLUMNS
    ]
    all_columns = COLUMNS + extra_columns

    new_frame = pd.DataFrame(valid_rows, columns=COLUMNS)
    for column in extra_columns:
        new_frame[column] = None

    if existing.empty:
        existing = pd.DataFrame(columns=all_columns)
    else:
        existing = existing.reindex(columns=all_columns)

    new_frame = new_frame.reindex(columns=all_columns)

    combined = pd.concat([existing, new_frame], ignore_index=True)
    row_count_before_dedup = len(combined)

    combined.drop_duplicates(
        subset="context_snapshot_id",
        keep="first",
        inplace=True,
    )

    row_count_after_dedup = len(combined)

    summary["inserted"] = int(
        row_count_after_dedup - len(existing)
    )
    summary["duplicates"] = int(
        row_count_before_dedup - row_count_after_dedup
    )

    try:
        combined = combined.reindex(columns=all_columns)
        combined.to_csv(
            CONTEXT_STORE_FILE,
            index=False,
            encoding="utf-8",
        )
        summary["stored_rows"] = int(row_count_after_dedup)
    except Exception as exc:
        summary["errors"].append(
            f"Failed to write context store: {exc}"
        )

    return summary


def read_context_history(
    game_id: Optional[Any] = None,
    pregame_only: bool = False,
    ready_only: bool = False,
) -> pd.DataFrame:
    """Read context history with optional filters."""
    frame = _load_context_store()

    if frame.empty:
        return frame

    if game_id is not None:
        frame = frame[frame["game_id"] == str(game_id)]

    if pregame_only and "is_pregame" in frame.columns:
        frame = frame[frame["is_pregame"] == True]

    if ready_only and "context_ready_for_betting" in frame.columns:
        frame = frame[frame["context_ready_for_betting"] == True]

    return frame.reset_index(drop=True)


def get_latest_pregame_context(
    game_id: Any,
) -> Optional[Dict[str, Any]]:
    """Return the newest valid pregame context snapshot for one game."""
    frame = read_context_history(
        game_id=game_id,
        pregame_only=True,
        ready_only=False,
    )

    if frame.empty:
        return None

    frame = frame.copy()
    frame["captured_at_parsed"] = pd.to_datetime(
        frame["captured_at"],
        errors="coerce",
        utc=True,
    )
    frame = frame.dropna(subset=["captured_at_parsed"])

    if frame.empty:
        return None

    latest_index = frame["captured_at_parsed"].idxmax()
    latest = frame.loc[latest_index].drop(
        labels=["captured_at_parsed"]
    )

    return latest.to_dict()


def get_latest_ready_pregame_context(
    game_id: Any,
) -> Optional[Dict[str, Any]]:
    """Return the newest betting-ready pregame snapshot for one game."""
    frame = read_context_history(
        game_id=game_id,
        pregame_only=True,
        ready_only=True,
    )

    if frame.empty:
        return None

    frame = frame.copy()
    frame["captured_at_parsed"] = pd.to_datetime(
        frame["captured_at"],
        errors="coerce",
        utc=True,
    )
    frame = frame.dropna(subset=["captured_at_parsed"])

    if frame.empty:
        return None

    latest_index = frame["captured_at_parsed"].idxmax()
    latest = frame.loc[latest_index].drop(
        labels=["captured_at_parsed"]
    )

    return latest.to_dict()
