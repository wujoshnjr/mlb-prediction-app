# prediction.py
"""Generate daily MLB predictions and write report/prediction.json.

The baseline pipeline is preserved while runtime contracts are tightened:
- ML inference uses the shared training feature schema.
- Empty schedules distinguish a real off-day from an upstream fetch failure.
- Experimental feature groups remain behind configuration switches.
- Runtime text is ASCII-only to avoid encoding damage during web edits.
- Suspicious odds markets are tracked but never used for bet recommendations.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import UnifiedSportsModel
from scripts.feature_schema import EXPECTED_FEATURES
from scripts.risk_guard import LiveBetRiskGuard

try:
    import config
except ImportError:
    class config:  # type: ignore[no-redef]
        RATINGS_ENGINE = "elo"
        FEATURE_USE_PITCH_MATCHUP = False
        FEATURE_USE_PITCH_USAGE = False
        ODDS_USE_CURVE_FEATURES = False
        NRFI_USE_ML = False
        MODEL_META = "lr"
        MODEL_USE_MLP = False
        WALKFORWARD_STRICT = False
        PIPELINE_VERSION = "baseline_v2_clean"
        SNAPSHOT_POLICY = "first_seen_pregame"
        BETTING_MODE = "paper_trading"
        MIN_MONEYLINE_EDGE = 0.03
        MAX_KELLY_FRACTION = 0.025


def optional_import(module_name: str, *names: str) -> tuple[Any | None, ...]:
    """Import optional helpers without aborting baseline prediction."""
    try:
        module = importlib.import_module(module_name)
        return tuple(getattr(module, name) for name in names)
    except Exception as exc:
        print(f"Optional import failed for {module_name}: {exc}")
        return tuple(None for _ in names)


(MLBElosystem,) = optional_import("scripts.elo", "MLBElosystem")
(MonteCarloSimulator,) = optional_import("scripts.monte_carlo", "MonteCarloSimulator")
(calculate_catcher_effect,) = optional_import(
    "scripts.catcher_utils", "calculate_catcher_effect"
)
(calculate_lag_features,) = optional_import(
    "scripts.lag_features", "calculate_lag_features"
)
(calculate_pitcher_ratings,) = optional_import(
    "scripts.player_ratings", "calculate_pitcher_ratings"
)
(calculate_bullpen_availability,) = optional_import(
    "scripts.bullpen_availability", "calculate_bullpen_availability"
)
(get_elo_momentum,) = optional_import("scripts.elo_momentum", "get_elo_momentum")
(get_pitch_type_matchup_score,) = optional_import(
    "scripts.pitch_type_matchup", "get_pitch_type_matchup_score"
)
(get_bradley_terry_strengths,) = optional_import(
    "scripts.bradley_terry", "get_bradley_terry_strengths"
)
(compute_pitch_usage_features,) = optional_import(
    "scripts.pitch_usage", "compute_pitch_usage_features"
)
(NRFIModel, extract_nrf_features) = optional_import(
    "scripts.nrf_model", "NRFIModel", "extract_nrf_features"
)
(load_glicko2_league,) = optional_import(
    "scripts.rating_updater", "load_glicko2_league"
)
(get_park_factor,) = optional_import("scripts.park_factors", "get_park_factor")
(append_first_seen_pregame_snapshots,) = optional_import(
    "scripts.snapshot_store",
    "append_first_seen_pregame_snapshots",
)
(append_game_market_snapshot, refresh_opening_closing_flags) = optional_import(
    "scripts.market_odds_store",
    "append_game_market_snapshot",
    "refresh_opening_closing_flags",
)

REPORT_FILE = Path("report/prediction.json")
HISTORY_FILE = Path("data/historical_predictions.csv")
HISTORICAL_DIR = Path("data/historical")
LAST_GAME_FILE = Path("data/team_last_game.json")
MODEL_FILE = Path("data/calibrator.pkl")
DAILY_CONTEXT_FILE = Path("data/daily_game_context.csv")
PROJECTED_LINEUP_CONTEXT_FILE = Path("data/projected_lineup_context.csv")
SAVANT_TOP3_CONTEXT_FILE = Path("data/savant_top3_context.csv")
WEATHER_CONTEXT_FILE = Path("data/weather_context.csv")
PITCHER_ADVANCED_CONTEXT_FILE = Path("data/pitcher_advanced_context.csv")
CONTEXT_FEATURE_BRIDGE_FILE = Path("data/context_feature_bridge.csv")
TEAM_FORM_CONTEXT_FILE = Path("data/team_form_context.csv")
RISK_GUARD = LiveBetRiskGuard(
    market_research_report_path="report/market_edge_research.json"
)

TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "D-backs",
    "Diamondbacks": "D-backs",
    "Arizona": "D-backs",
    "Atlanta Braves": "Braves",
    "Atlanta": "Braves",
    "Baltimore Orioles": "Orioles",
    "Baltimore": "Orioles",
    "Boston Red Sox": "Red Sox",
    "Boston": "Red Sox",
    "Chicago Cubs": "Cubs",
    "Chicago (NL)": "Cubs",
    "Chicago White Sox": "White Sox",
    "Chicago (AL)": "White Sox",
    "Cincinnati Reds": "Reds",
    "Cincinnati": "Reds",
    "Cleveland Guardians": "Guardians",
    "Cleveland": "Guardians",
    "Colorado Rockies": "Rockies",
    "Colorado": "Rockies",
    "Detroit Tigers": "Tigers",
    "Detroit": "Tigers",
    "Houston Astros": "Astros",
    "Houston": "Astros",
    "Kansas City Royals": "Royals",
    "Kansas City": "Royals",
    "Los Angeles Angels": "Angels",
    "Los Angeles (AL)": "Angels",
    "Los Angeles Dodgers": "Dodgers",
    "Los Angeles (NL)": "Dodgers",
    "Miami Marlins": "Marlins",
    "Miami": "Marlins",
    "Milwaukee Brewers": "Brewers",
    "Milwaukee": "Brewers",
    "Minnesota Twins": "Twins",
    "Minnesota": "Twins",
    "New York Mets": "Mets",
    "New York (NL)": "Mets",
    "New York Yankees": "Yankees",
    "New York (AL)": "Yankees",
    "Oakland Athletics": "Athletics",
    "Oakland": "Athletics",
    "Athletics": "Athletics",
    "Philadelphia Phillies": "Phillies",
    "Philadelphia": "Phillies",
    "Pittsburgh Pirates": "Pirates",
    "Pittsburgh": "Pirates",
    "San Diego Padres": "Padres",
    "San Diego": "Padres",
    "San Francisco Giants": "Giants",
    "San Francisco": "Giants",
    "Seattle Mariners": "Mariners",
    "Seattle": "Mariners",
    "St. Louis Cardinals": "Cardinals",
    "St. Louis": "Cardinals",
    "Tampa Bay Rays": "Rays",
    "Tampa Bay": "Rays",
    "Texas Rangers": "Rangers",
    "Texas": "Rangers",
    "Toronto Blue Jays": "Blue Jays",
    "Toronto": "Blue Jays",
    "Washington Nationals": "Nationals",
    "Washington": "Nationals",
}

TEAM_ID_MAP = {
    "Braves": 144,
    "Orioles": 110,
    "Red Sox": 111,
    "Cubs": 112,
    "White Sox": 145,
    "Reds": 113,
    "Guardians": 114,
    "Rockies": 115,
    "Tigers": 116,
    "Astros": 117,
    "Royals": 118,
    "Angels": 108,
    "Dodgers": 119,
    "Marlins": 146,
    "Brewers": 158,
    "Twins": 142,
    "Mets": 121,
    "Yankees": 147,
    "Athletics": 133,
    "Phillies": 143,
    "Pirates": 134,
    "Padres": 135,
    "Giants": 137,
    "Mariners": 136,
    "Cardinals": 138,
    "Rays": 139,
    "Rangers": 140,
    "Blue Jays": 141,
    "Nationals": 120,
    "D-backs": 109,
}

TEAM_TIMEZONES = {
    "Braves": "Eastern",
    "Orioles": "Eastern",
    "Red Sox": "Eastern",
    "Cubs": "Central",
    "White Sox": "Central",
    "Reds": "Eastern",
    "Guardians": "Eastern",
    "Rockies": "Mountain",
    "Tigers": "Eastern",
    "Astros": "Central",
    "Royals": "Central",
    "Angels": "Pacific",
    "Dodgers": "Pacific",
    "Marlins": "Eastern",
    "Brewers": "Central",
    "Twins": "Central",
    "Mets": "Eastern",
    "Yankees": "Eastern",
    "Athletics": "Pacific",
    "Phillies": "Eastern",
    "Pirates": "Eastern",
    "Padres": "Pacific",
    "Giants": "Pacific",
    "Mariners": "Pacific",
    "Cardinals": "Central",
    "Rays": "Eastern",
    "Rangers": "Central",
    "Blue Jays": "Eastern",
    "Nationals": "Eastern",
    "D-backs": "Mountain",
}

TIMEZONE_OFFSETS = {
    "Eastern": 0,
    "Central": -1,
    "Mountain": -2,
    "Pacific": -3,
}


def normalize_team(team: Any) -> str:
    text = "" if team is None else str(team).strip()
    return TEAM_NAME_MAP.get(text, text)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def is_missing_value(value: Any) -> bool:
    """Return True when a scalar report value should be treated as missing."""
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


def as_optional_float(value: Any) -> float | None:
    """Return a float or None without converting missing values to zero."""
    if is_missing_value(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_optional_bool(value: Any) -> bool | None:
    """Normalize optional bool values stored in CSV or pandas rows."""
    if is_missing_value(value):
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


def parse_json_list(value: Any) -> list[Any]:
    """Parse a JSON list value safely."""
    if is_missing_value(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (TypeError, ValueError):
            return []

    return []


def parse_csv_int_list(value: Any) -> list[int]:
    """Parse comma-separated player IDs such as '111,222,333' safely."""
    if is_missing_value(value):
        return []

    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value).split(",")

    result: list[int] = []
    for item in raw_items:
        try:
            text = str(item).strip()
            if not text or text.lower() in {"nan", "none", "null"}:
                continue
            result.append(int(float(text)))
        except (TypeError, ValueError):
            continue

    return result


def as_optional_int(value: Any) -> int | None:
    """Return int or None without converting missing values to zero."""
    if is_missing_value(value):
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_optional_str(value: Any) -> str:
    """Return a clean string, treating None/NaN/'nan' as empty."""
    if is_missing_value(value):
        return ""

    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""

    return text


def calculate_context_weather_features(
    context_row: dict[str, Any],
) -> dict[str, float]:
    """Build conservative weather features from daily game context.

    These are intentionally small feature values. They are feature inputs, not
    direct probability adjustments.
    """
    weather_temp = as_optional_float(context_row.get("weather_temp"))
    wind_speed = as_optional_float(context_row.get("wind_speed"))
    wind_direction = str(context_row.get("wind_direction") or "").lower()
    weather_condition = str(context_row.get("weather_condition") or "").lower()

    temp_effect = 0.0
    wind_effect = 0.0
    precip_effect = 0.0

    if weather_temp is not None:
        # 72F is treated as neutral. Clamp to avoid unstable effects.
        temp_effect = float(np.clip((weather_temp - 72.0) / 100.0, -0.03, 0.03))

    if wind_speed is not None and wind_speed > 0:
        raw_wind = float(np.clip(wind_speed / 400.0, 0.0, 0.03))

        if any(token in wind_direction for token in ("out", "to cf", "to lf", "to rf")):
            wind_effect = raw_wind
        elif any(token in wind_direction for token in ("in", "from cf", "from lf", "from rf")):
            wind_effect = -raw_wind

    if any(token in weather_condition for token in ("rain", "drizzle", "storm")):
        precip_effect = -0.01

    return {
        "temp_effect": temp_effect,
        "wind_effect": wind_effect,
        "precip_effect": precip_effect,
    }


def merge_projected_lineup_context(
    frame: pd.DataFrame,
    errors: list[str],
) -> pd.DataFrame:
    """Merge conservative projected lineup context into latest daily context rows."""
    if frame.empty or not PROJECTED_LINEUP_CONTEXT_FILE.exists():
        return frame

    try:
        projected = pd.read_csv(PROJECTED_LINEUP_CONTEXT_FILE)
    except Exception as exc:
        errors.append(f"Unable to read projected lineup context: {exc}")
        return frame

    if projected.empty or "game_id" not in projected.columns:
        return frame

    projected = projected.copy()
    projected["game_id"] = projected["game_id"].astype(str)

    if "projected_lineup_captured_at" in projected.columns:
        projected["projected_lineup_captured_at_dt"] = pd.to_datetime(
            projected["projected_lineup_captured_at"],
            errors="coerce",
            utc=True,
        )
        if projected["projected_lineup_captured_at_dt"].notna().any():
            projected = projected.sort_values("projected_lineup_captured_at_dt")
            projected = projected.groupby("game_id", as_index=False).tail(1)

    keep_columns = [
        "game_id",
        "home_projected_lineup_available",
        "away_projected_lineup_available",
        "home_projected_lineup_status",
        "away_projected_lineup_status",
        "home_projected_player_count",
        "away_projected_player_count",
        "home_projected_top3_player_ids",
        "away_projected_top3_player_ids",
        "home_projected_lineup_reason",
        "away_projected_lineup_reason",
        "projected_lineup_source",
        "projected_lineup_captured_at",
    ]
    keep_columns = [column for column in keep_columns if column in projected.columns]

    merged = frame.copy()
    merged["game_id"] = merged["game_id"].astype(str)
    projected = projected[keep_columns].copy()

    merged = merged.merge(
        projected,
        on="game_id",
        how="left",
        suffixes=("", "_projected_source"),
    )

    for column in (
        "home_projected_top3_player_ids",
        "away_projected_top3_player_ids",
        "home_projected_lineup_status",
        "away_projected_lineup_status",
        "home_projected_player_count",
        "away_projected_player_count",
        "home_projected_lineup_reason",
        "away_projected_lineup_reason",
        "projected_lineup_source",
        "projected_lineup_captured_at",
    ):
        source_column = f"{column}_projected_source"
        if source_column in merged.columns:
            if column in merged.columns:
                merged[column] = merged[column].where(
                    merged[column].notna(),
                    merged[source_column],
                )
            else:
                merged[column] = merged[source_column]
            merged.drop(columns=[source_column], inplace=True)

    for column in (
        "home_projected_lineup_available",
        "away_projected_lineup_available",
    ):
        source_column = f"{column}_projected_source"
        if source_column in merged.columns:
            if column in merged.columns:
                merged[column] = merged[column].where(
                    merged[column].notna(),
                    merged[source_column],
                )
            else:
                merged[column] = merged[source_column]
            merged.drop(columns=[source_column], inplace=True)

    return merged


def latest_savant_top3_context_by_game(
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    """Load latest Savant top-3 hitter context by game_id."""
    if not SAVANT_TOP3_CONTEXT_FILE.exists():
        return {}

    try:
        frame = pd.read_csv(SAVANT_TOP3_CONTEXT_FILE)
    except Exception as exc:
        errors.append(f"Unable to read Savant top3 context: {exc}")
        return {}

    if frame.empty or "game_id" not in frame.columns:
        return {}

    frame = frame.copy()
    frame["game_id"] = frame["game_id"].astype(str)

    if "captured_at" in frame.columns:
        frame["captured_at_parsed"] = pd.to_datetime(
            frame["captured_at"],
            errors="coerce",
            utc=True,
        )
        if frame["captured_at_parsed"].notna().any():
            frame = frame.sort_values("captured_at_parsed")
            frame = frame.groupby("game_id", as_index=False).tail(1)

    return {
        str(row["game_id"]): row.to_dict()
        for _, row in frame.iterrows()
    }


def latest_weather_context_by_game(
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    """Load latest weather context by game_id."""
    if not WEATHER_CONTEXT_FILE.exists():
        return {}

    try:
        frame = pd.read_csv(WEATHER_CONTEXT_FILE)
    except Exception as exc:
        errors.append(f"Unable to read weather context: {exc}")
        return {}

    if frame.empty or "game_id" not in frame.columns:
        return {}

    frame = frame.copy()
    frame["game_id"] = frame["game_id"].astype(str)

    if "weather_captured_at" in frame.columns:
        frame["weather_captured_at_parsed"] = pd.to_datetime(
            frame["weather_captured_at"],
            errors="coerce",
            utc=True,
        )
        if frame["weather_captured_at_parsed"].notna().any():
            frame = frame.sort_values("weather_captured_at_parsed")
            frame = frame.groupby("game_id", as_index=False).tail(1)

    return {
        str(row["game_id"]): row.to_dict()
        for _, row in frame.iterrows()
    }


def build_weather_context_summary(
    weather_row: dict[str, Any],
) -> dict[str, Any]:
    """Return compact weather context summary for prediction report."""
    if not weather_row:
        return {
            "available": False,
            "weather_source_status": "missing",
            "weather_reason": "No weather_context row matched this game_id",
        }

    return {
        "available": True,
        "weather_source": str(weather_row.get("weather_source", "")),
        "weather_source_status": str(weather_row.get("weather_source_status", "")),
        "weather_captured_at": str(weather_row.get("weather_captured_at", "")),
        "weather_forecast_time": str(weather_row.get("weather_forecast_time", "")),
        "venue_name": str(weather_row.get("venue_name", "")),
        "weather_is_dome": bool(weather_row.get("weather_is_dome", False)),
        "weather_temp_f": as_float(weather_row.get("weather_temp_f"), None),
        "weather_wind_speed_mph": as_float(
            weather_row.get("weather_wind_speed_mph"),
            None,
        ),
        "weather_precip_probability": as_float(
            weather_row.get("weather_precip_probability"),
            None,
        ),
        "weather_condition": str(weather_row.get("weather_condition", "")),
        "temp_effect": as_float(weather_row.get("temp_effect"), 0.0),
        "wind_effect": as_float(weather_row.get("wind_effect"), 0.0),
        "precip_effect": as_float(weather_row.get("precip_effect"), 0.0),
        "weather_reason": str(weather_row.get("weather_reason", "")),
    }


def latest_context_csv_by_game(
    path: Path,
    captured_at_column: str,
    errors: list[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    """Load latest generic context CSV rows by game_id."""
    if not path.exists():
        return {}

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"Unable to read {label}: {exc}")
        return {}

    if frame.empty or "game_id" not in frame.columns:
        return {}

    frame = frame.copy()
    frame["game_id"] = frame["game_id"].astype(str)

    if captured_at_column in frame.columns:
        parsed_column = f"{captured_at_column}_parsed"
        frame[parsed_column] = pd.to_datetime(
            frame[captured_at_column],
            errors="coerce",
            utc=True,
        )
        if frame[parsed_column].notna().any():
            frame = frame.sort_values(parsed_column)
            frame = frame.groupby("game_id", as_index=False).tail(1)

    return {
        str(row["game_id"]): row.to_dict()
        for _, row in frame.iterrows()
    }


def latest_daily_context_by_game(
    errors: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load the latest pregame daily context snapshot for each game."""
    summary: dict[str, Any] = {
        "status": "unavailable",
        "file": str(DAILY_CONTEXT_FILE),
        "stored_rows": 0,
        "latest_context_count": 0,
        "ready_context_count": 0,
        "errors": [],
    }

    if not DAILY_CONTEXT_FILE.exists():
        summary["reason"] = "daily_context_file_missing"
        return {}, summary

    try:
        frame = pd.read_csv(DAILY_CONTEXT_FILE)
    except Exception as exc:
        message = f"Unable to read daily context file: {exc}"
        errors.append(message)
        summary["status"] = "failed"
        summary["errors"].append(message)
        return {}, summary

    if frame.empty:
        summary["status"] = "empty"
        return {}, summary

    summary["stored_rows"] = int(len(frame))

    if "game_id" not in frame.columns or "captured_at" not in frame.columns:
        message = "Daily context file missing game_id or captured_at."
        errors.append(message)
        summary["status"] = "failed"
        summary["errors"].append(message)
        return {}, summary

    frame = frame.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame = merge_projected_lineup_context(frame, errors)

    if "is_pregame" in frame.columns:
        frame = frame[
            frame["is_pregame"].apply(as_optional_bool) == True
        ]

    frame["captured_at_parsed"] = pd.to_datetime(
        frame["captured_at"],
        errors="coerce",
        utc=True,
    )
    frame = frame.dropna(subset=["captured_at_parsed"])

    if frame.empty:
        summary["status"] = "empty_after_filter"
        return {}, summary

    frame.sort_values("captured_at_parsed", inplace=True)
    latest = frame.groupby("game_id", as_index=False).tail(1)

    context_by_game: dict[str, dict[str, Any]] = {
        str(row["game_id"]): row.to_dict()
        for _, row in latest.iterrows()
    }

    ready_count = 0
    if "context_ready_for_betting" in latest.columns:
        ready_count = int(
            latest["context_ready_for_betting"]
            .apply(as_optional_bool)
            .fillna(False)
            .sum()
        )

    summary.update(
        {
            "status": "completed",
            "latest_context_count": int(len(context_by_game)),
            "ready_context_count": ready_count,
        }
    )

    return context_by_game, summary


def build_daily_context_summary(
    game_id: Any,
    context_by_game: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact context status block for one prediction item."""
    row = context_by_game.get(str(game_id))

    if row is None:
        return {
            "status": "CONTEXT_UNAVAILABLE",
            "context_ready_for_betting": False,
            "data_completeness_score": None,
            "captured_at": None,
            "missing_critical_fields": [],
            "pitcher_status": "missing",
            "lineup_status": "missing",
            "bullpen_status": "missing",
            "closer_status": "unknown",
            "signal_status": "tracking_only_no_context",
        }

    home_sp_confirmed = as_optional_bool(
        row.get("home_starting_pitcher_confirmed")
    )
    away_sp_confirmed = as_optional_bool(
        row.get("away_starting_pitcher_confirmed")
    )
    home_lineup_confirmed = as_optional_bool(
        row.get("home_lineup_confirmed")
    )
    away_lineup_confirmed = as_optional_bool(
        row.get("away_lineup_confirmed")
    )

    home_lineup_count = as_int(row.get("home_lineup_player_count"), 0)
    away_lineup_count = as_int(row.get("away_lineup_player_count"), 0)

    home_lineup_ids = parse_json_list(row.get("home_lineup_player_ids_json"))
    away_lineup_ids = parse_json_list(row.get("away_lineup_player_ids_json"))

    if home_lineup_count <= 0:
        home_lineup_count = len(home_lineup_ids)
    if away_lineup_count <= 0:
        away_lineup_count = len(away_lineup_ids)

    home_top3_player_ids = parse_csv_int_list(
        row.get("home_top3_player_ids")
        or row.get("home_projected_top3_player_ids")
    )
    away_top3_player_ids = parse_csv_int_list(
        row.get("away_top3_player_ids")
        or row.get("away_projected_top3_player_ids")
    )

    home_projected_lineup_available = (
        home_lineup_count >= 7
        or as_optional_bool(row.get("home_projected_lineup_available")) is True
    )
    away_projected_lineup_available = (
        away_lineup_count >= 7
        or as_optional_bool(row.get("away_projected_lineup_available")) is True
    )

    home_projected_lineup_status = as_optional_str(
        row.get("home_projected_lineup_status")
    )
    away_projected_lineup_status = as_optional_str(
        row.get("away_projected_lineup_status")
    )

    home_top3_available = (
        home_projected_lineup_status == "projected_top3_available"
        or len(home_top3_player_ids) >= 3
    )
    away_top3_available = (
        away_projected_lineup_status == "projected_top3_available"
        or len(away_top3_player_ids) >= 3
    )

    game_feed_available = as_optional_bool(row.get("game_feed_available")) is True

    home_starter_status = as_optional_str(row.get("home_starter_status"))
    away_starter_status = as_optional_str(row.get("away_starter_status"))
    home_starter_confidence = as_optional_bool(
        row.get("home_starter_confidence")
    )
    away_starter_confidence = as_optional_bool(
        row.get("away_starter_confidence")
    )
    home_starter_confidence_score = as_optional_float(
        row.get("home_starter_confidence_score")
    )
    away_starter_confidence_score = as_optional_float(
        row.get("away_starter_confidence_score")
    )
    home_starter_reason = as_optional_str(row.get("home_starter_reason"))
    away_starter_reason = as_optional_str(row.get("away_starter_reason"))
    
    home_starting_pitcher_id = as_optional_int(
        row.get("home_starting_pitcher_id")
    )
    away_starting_pitcher_id = as_optional_int(
        row.get("away_starting_pitcher_id")
    )

    weather_temp = as_optional_float(row.get("weather_temp"))
    wind_speed = as_optional_float(row.get("wind_speed"))
    weather_condition = as_optional_str(row.get("weather_condition"))
    wind_direction = as_optional_str(row.get("wind_direction"))

    umpire_home_plate_id = as_optional_int(row.get("umpire_home_plate_id"))
    umpire_home_plate_name = as_optional_str(row.get("umpire_home_plate_name"))
    home_bullpen_available = as_optional_bool(
        row.get("home_bullpen_data_available")
    )
    away_bullpen_available = as_optional_bool(
        row.get("away_bullpen_data_available")
    )
    home_closer_available_known = as_optional_bool(
        row.get("home_closer_available_known")
    )
    away_closer_available_known = as_optional_bool(
        row.get("away_closer_available_known")
    )
    home_closer_available = as_optional_bool(row.get("home_closer_available"))
    away_closer_available = as_optional_bool(row.get("away_closer_available"))

    home_closer_known = home_closer_available_known is True
    away_closer_known = away_closer_available_known is True

    home_closer_status = as_optional_str(row.get("home_closer_status"))
    away_closer_status = as_optional_str(row.get("away_closer_status"))
    home_closer_risk_score = as_optional_float(row.get("home_closer_risk_score"))
    away_closer_risk_score = as_optional_float(row.get("away_closer_risk_score"))
    home_closer_reason = as_optional_str(row.get("home_closer_reason"))
    away_closer_reason = as_optional_str(row.get("away_closer_reason"))

    context_ready = as_optional_bool(
        row.get("context_ready_for_betting")
    ) is True

    home_starter_high_confidence = (
        home_starter_status in {"confirmed", "high_confidence_probable"}
        or home_starter_confidence is True
    )
    away_starter_high_confidence = (
        away_starter_status in {"confirmed", "high_confidence_probable"}
        or away_starter_confidence is True
    )

    if home_sp_confirmed is True and away_sp_confirmed is True:
        pitcher_status = "confirmed"
    elif home_starter_high_confidence and away_starter_high_confidence:
        pitcher_status = "high_confidence_probable"
    elif home_starter_high_confidence or away_starter_high_confidence:
        pitcher_status = "mixed_probable"
    elif (
        not is_missing_value(row.get("home_probable_pitcher_id"))
        or not is_missing_value(row.get("away_probable_pitcher_id"))
    ):
        pitcher_status = "low_confidence_probable"
    else:
        pitcher_status = "missing"

    starter_confidence_status = (
        "known"
        if home_starter_status and away_starter_status
        else "partial"
        if home_starter_status or away_starter_status
        else "unknown"
    )

    if home_lineup_confirmed is True and away_lineup_confirmed is True:
        lineup_status = "confirmed"
    elif home_lineup_confirmed is True or away_lineup_confirmed is True:
        lineup_status = "partial_confirmed"
    elif home_projected_lineup_available and away_projected_lineup_available:
        lineup_status = "projected_available"
    else:
        lineup_status = "pending"

    lineup_projection_status = (
        "available"
        if home_projected_lineup_available and away_projected_lineup_available
        else "partial"
        if home_projected_lineup_available or away_projected_lineup_available
        else "top3_only"
        if home_top3_available and away_top3_available
        else "partial_top3"
        if home_top3_available or away_top3_available
        else "missing"
    )

    if home_bullpen_available is True and away_bullpen_available is True:
        bullpen_status = "available"
    elif home_bullpen_available is True or away_bullpen_available is True:
        bullpen_status = "partial"
    else:
        bullpen_status = "missing"

    closer_status = (
        "known"
        if home_closer_known and away_closer_known
        else "unknown"
    )

    raw_missing_critical_fields = parse_json_list(
        row.get("missing_critical_fields_json")
    )

    missing_critical_fields = list(raw_missing_critical_fields)
    starter_confirmation_pending = False

    if pitcher_status in {"confirmed", "high_confidence_probable"}:
        starter_fields = {
            "home_starting_pitcher_confirmed",
            "away_starting_pitcher_confirmed",
        }
        starter_confirmation_pending = any(
            field in missing_critical_fields for field in starter_fields
        )
        missing_critical_fields = [
            field
            for field in missing_critical_fields
            if field not in starter_fields
        ]

    if missing_critical_fields:
        context_not_ready_reason = (
            "Missing critical fields: "
            + ", ".join(str(field) for field in missing_critical_fields)
        )
    elif starter_confirmation_pending:
        context_not_ready_reason = (
            "Official starter confirmation pending, but high-confidence probable starters are available."
        )
    else:
        context_not_ready_reason = row.get("context_not_ready_reason")

    return {
        "status": (
            "READY_FOR_BETTING"
            if context_ready
            else "TRACKING_ONLY_CONTEXT_NOT_READY"
        ),
        "context_snapshot_id": row.get("context_snapshot_id"),
        "captured_at": row.get("captured_at"),
        "data_completeness_score": as_optional_float(
            row.get("data_completeness_score")
        ),
        "context_ready_for_betting": context_ready,
        "context_not_ready_reason": context_not_ready_reason,
        "raw_context_not_ready_reason": row.get("context_not_ready_reason"),
        "missing_critical_fields": missing_critical_fields,
        "raw_missing_critical_fields": raw_missing_critical_fields,
        "pitcher_status": pitcher_status,
        "starter_confidence_status": starter_confidence_status,
        "starter_confirmation_pending": starter_confirmation_pending,
        "home_starter_status": home_starter_status,
        "away_starter_status": away_starter_status,
        "home_starter_confidence": home_starter_confidence,
        "away_starter_confidence": away_starter_confidence,
        "home_starter_confidence_score": home_starter_confidence_score,
        "away_starter_confidence_score": away_starter_confidence_score,
        "home_starter_reason": home_starter_reason,
        "away_starter_reason": away_starter_reason,
        "lineup_status": lineup_status,
        "lineup_projection_status": lineup_projection_status,
        "home_projected_lineup_available": home_projected_lineup_available,
        "away_projected_lineup_available": away_projected_lineup_available,
        "home_projected_lineup_status": home_projected_lineup_status,
        "away_projected_lineup_status": away_projected_lineup_status,
        "home_top3_available": home_top3_available,
        "away_top3_available": away_top3_available,
        "home_projected_lineup_reason": as_optional_str(
            row.get("home_projected_lineup_reason")
        ),
        "away_projected_lineup_reason": as_optional_str(
            row.get("away_projected_lineup_reason")
        ),
        "bullpen_status": bullpen_status,
        "closer_status": closer_status,
        "home_closer_available_known": home_closer_available_known,
        "away_closer_available_known": away_closer_available_known,
        "home_closer_available": home_closer_available,
        "away_closer_available": away_closer_available,
        "home_closer_status": home_closer_status,
        "away_closer_status": away_closer_status,
        "home_closer_risk_score": home_closer_risk_score,
        "away_closer_risk_score": away_closer_risk_score,
        "home_closer_reason": home_closer_reason,
        "away_closer_reason": away_closer_reason,
        "signal_status": (
            "eligible_context"
            if context_ready
            else "tracking_only_context_incomplete"
        ),
        "home_probable_pitcher_id": row.get("home_probable_pitcher_id"),
        "home_probable_pitcher_name": as_optional_str(
            row.get("home_probable_pitcher_name")
        ),
        "away_probable_pitcher_id": row.get("away_probable_pitcher_id"),
        "away_probable_pitcher_name": as_optional_str(
            row.get("away_probable_pitcher_name")
        ),
        "home_lineup_confirmed": home_lineup_confirmed,
        "away_lineup_confirmed": away_lineup_confirmed,
        "home_lineup_player_count": home_lineup_count,
        "away_lineup_player_count": away_lineup_count,
        "game_feed_available": game_feed_available,
        "home_starting_pitcher_id": home_starting_pitcher_id,
        "away_starting_pitcher_id": away_starting_pitcher_id,
        "home_starting_pitcher_name": as_optional_str(
            row.get("home_starting_pitcher_name")
        ),
        "away_starting_pitcher_name": as_optional_str(
            row.get("away_starting_pitcher_name")
        ),
        "home_top3_player_ids": home_top3_player_ids,
        "away_top3_player_ids": away_top3_player_ids,
        "home_catcher_id": as_optional_int(row.get("home_catcher_id")),
        "away_catcher_id": as_optional_int(row.get("away_catcher_id")),
        "home_catcher_name": as_optional_str(row.get("home_catcher_name")),
        "away_catcher_name": as_optional_str(row.get("away_catcher_name")),
        "weather_temp": weather_temp,
        "weather_condition": weather_condition,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "umpire_home_plate_id": umpire_home_plate_id,
        "umpire_home_plate_name": umpire_home_plate_name,
        "home_bullpen_data_available": home_bullpen_available,
        "away_bullpen_data_available": away_bullpen_available,
        "home_bullpen_pitches_last_1d": as_optional_float(
            row.get("home_bullpen_pitches_last_1d")
        ),
        "away_bullpen_pitches_last_1d": as_optional_float(
            row.get("away_bullpen_pitches_last_1d")
        ),
        "home_bullpen_pitches_last_3d": as_optional_float(
            row.get("home_bullpen_pitches_last_3d")
        ),
        "away_bullpen_pitches_last_3d": as_optional_float(
            row.get("away_bullpen_pitches_last_3d")
        ),
        "home_bullpen_fatigue_score": as_optional_float(
            row.get("home_bullpen_fatigue_score")
        ),
        "away_bullpen_fatigue_score": as_optional_float(
            row.get("away_bullpen_fatigue_score")
        ),
    }


def summarize_prediction_context(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize daily context and betting readiness across prediction rows."""
    total = len(predictions)
    with_context = 0
    ready = 0
    bullpen_available = 0
    closer_known = 0
    starter_high_confidence = 0
    lineup_confirmed = 0
    lineup_projected_available = 0
    tracking_only = 0

    official_ready_count = 0
    practical_ready_count = 0
    risk_blocked_count = 0
    live_bet_candidate_count = 0
    betting_readiness_score_sum = 0.0
    betting_readiness_score_count = 0

    pitcher_status_counts: dict[str, int] = {}
    lineup_status_counts: dict[str, int] = {}
    signal_status_counts: dict[str, int] = {}
    betting_readiness_status_counts: dict[str, int] = {}
    risk_flag_counts: dict[str, int] = {}

    for item in predictions:
        summary = item.get("daily_context_summary", {})
        if summary.get("status") != "CONTEXT_UNAVAILABLE":
            with_context += 1

        if summary.get("context_ready_for_betting") is True:
            ready += 1
        else:
            tracking_only += 1

        if summary.get("bullpen_status") == "available":
            bullpen_available += 1

        if summary.get("closer_status") == "known":
            closer_known += 1

        if summary.get("pitcher_status") in {
            "confirmed",
            "high_confidence_probable",
        }:
            starter_high_confidence += 1

        if summary.get("lineup_status") == "confirmed":
            lineup_confirmed += 1

        if summary.get("lineup_projection_status") == "available":
            lineup_projected_available += 1

        pitcher_status = str(summary.get("pitcher_status") or "unknown")
        lineup_status = str(summary.get("lineup_status") or "unknown")
        signal_status = str(summary.get("signal_status") or "unknown")

        pitcher_status_counts[pitcher_status] = (
            pitcher_status_counts.get(pitcher_status, 0) + 1
        )
        lineup_status_counts[lineup_status] = (
            lineup_status_counts.get(lineup_status, 0) + 1
        )
        signal_status_counts[signal_status] = (
            signal_status_counts.get(signal_status, 0) + 1
        )

        readiness = item.get("betting_readiness") or {}
        readiness_status = str(
            item.get("betting_readiness_status")
            or readiness.get("betting_readiness_status")
            or "unknown"
        )
        betting_readiness_status_counts[readiness_status] = (
            betting_readiness_status_counts.get(readiness_status, 0) + 1
        )

        if readiness_status == "official_ready":
            official_ready_count += 1
        if readiness_status == "practical_ready":
            practical_ready_count += 1
        if readiness_status == "risk_blocked":
            risk_blocked_count += 1

        if item.get("live_bet_candidate") is True:
            live_bet_candidate_count += 1

        score = item.get("betting_readiness_score")
        if score is None:
            score = readiness.get("betting_readiness_score")
        score_float = as_optional_float(score)
        if score_float is not None:
            betting_readiness_score_sum += score_float
            betting_readiness_score_count += 1

        flags = item.get("betting_risk_flags")
        if flags is None:
            flags = readiness.get("betting_risk_flags")
        if isinstance(flags, list):
            for flag in flags:
                key = str(flag)
                risk_flag_counts[key] = risk_flag_counts.get(key, 0) + 1

    average_score = (
        round(betting_readiness_score_sum / betting_readiness_score_count, 4)
        if betting_readiness_score_count > 0
        else None
    )

    return {
        "prediction_count": total,
        "with_context_count": with_context,
        "ready_context_count": ready,
        "tracking_only_context_count": tracking_only,
        "bullpen_available_count": bullpen_available,
        "closer_known_count": closer_known,
        "starter_high_confidence_count": starter_high_confidence,
        "confirmed_lineup_context_count": lineup_confirmed,
        "projected_lineup_available_count": lineup_projected_available,
        "official_ready_count": official_ready_count,
        "practical_ready_count": practical_ready_count,
        "risk_blocked_count": risk_blocked_count,
        "live_bet_candidate_count": live_bet_candidate_count,
        "average_betting_readiness_score": average_score,
        "pitcher_status_counts": pitcher_status_counts,
        "lineup_status_counts": lineup_status_counts,
        "signal_status_counts": signal_status_counts,
        "betting_readiness_status_counts": betting_readiness_status_counts,
        "risk_flag_counts": risk_flag_counts,
    }


def _clamp_float(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp a numeric value to a closed interval."""
    return max(minimum, min(maximum, float(value)))


def evaluate_betting_readiness(
    daily_context_summary: dict[str, Any],
    *,
    odds_quality_status: str,
    moneyline_gate_status: str,
    model_source: str,
    model_edge_home: float | None,
    moneyline_selected_edge: float | None,
    min_moneyline_edge: float,
    model_training_sample_count: int | None = None,
    production_sample_threshold: int = 300,
) -> dict[str, Any]:
    """
    Evaluate effective betting readiness without changing model probability,
    odds logic, or recommendation status.

    This is a paper-trading readiness layer.
    """
    context = daily_context_summary or {}

    raw_context_ready = context.get("context_ready_for_betting") is True

    odds_status = str(odds_quality_status or "UNAVAILABLE").strip().upper()
    gate_status = str(moneyline_gate_status or "").strip().upper()
    model_source_norm = str(model_source or "").strip().lower()

    missing_critical_fields = context.get("missing_critical_fields") or []
    if not isinstance(missing_critical_fields, list):
        missing_critical_fields = []

    pitcher_status = str(context.get("pitcher_status") or "missing")
    starter_confidence_status = str(
        context.get("starter_confidence_status") or "unknown"
    )
    lineup_status = str(context.get("lineup_status") or "pending")
    bullpen_status = str(context.get("bullpen_status") or "missing")
    closer_status = str(context.get("closer_status") or "unknown")

    home_closer_status = str(context.get("home_closer_status") or "unknown")
    away_closer_status = str(context.get("away_closer_status") or "unknown")

    starter_confirmation_pending = (
        context.get("starter_confirmation_pending") is True
    )

    selected_edge = (
        moneyline_selected_edge
        if moneyline_selected_edge is not None
        else abs(model_edge_home)
        if model_edge_home is not None
        else None
    )

    risk_flags: list[str] = []
    reasons: list[str] = []

    if odds_status != "OK":
        risk_flags.append("odds_not_ok")
        reasons.append(f"odds_quality_status={odds_status}")

    if model_source_norm != "ml":
        risk_flags.append("non_ml_model")
        reasons.append(f"model_source={model_source_norm or 'unknown'}")

    if missing_critical_fields:
        risk_flags.append("context_missing_fields")
        reasons.append(
            "missing_critical_fields="
            + ",".join(str(item) for item in missing_critical_fields[:6])
        )

    if lineup_status != "confirmed":
        risk_flags.append("lineup_not_confirmed")
        reasons.append(f"lineup_status={lineup_status}")

    if bullpen_status != "available":
        risk_flags.append("bullpen_not_available")
        reasons.append(f"bullpen_status={bullpen_status}")

    if starter_confirmation_pending:
        risk_flags.append("starter_confirmation_pending")
        reasons.append("official starter confirmation pending")

    if (
        home_closer_status == "high_fatigue_risk"
        or away_closer_status == "high_fatigue_risk"
    ):
        risk_flags.append("closer_high_fatigue")
        reasons.append(
            f"closer_status home={home_closer_status}, away={away_closer_status}"
        )
    elif (
        home_closer_status == "fatigue_risk"
        or away_closer_status == "fatigue_risk"
    ):
        risk_flags.append("closer_fatigue")
        reasons.append(
            f"closer_status home={home_closer_status}, away={away_closer_status}"
        )

    if selected_edge is None:
        risk_flags.append("edge_unavailable")
        reasons.append("selected edge unavailable")
    elif selected_edge < min_moneyline_edge:
        risk_flags.append("edge_below_threshold")
        reasons.append(
            f"selected_edge={selected_edge:.4f} below threshold={min_moneyline_edge:.4f}"
        )

    sample_count = (
        int(model_training_sample_count)
        if model_training_sample_count is not None
        else 0
    )

    if sample_count < production_sample_threshold:
        risk_flags.append("early_model_small_sample")
        reasons.append(
            f"model_training_sample_count={sample_count} below production threshold={production_sample_threshold}"
        )

    selected_edge_abs = abs(float(selected_edge)) if selected_edge is not None else None
    large_edge_threshold = 0.08

    if (
        selected_edge_abs is not None
        and selected_edge_abs >= large_edge_threshold
        and sample_count < production_sample_threshold
    ):
        risk_flags.append("large_anti_market_edge_early_model")
        reasons.append(
            f"selected_edge_abs={selected_edge_abs:.4f} exceeds large-edge threshold={large_edge_threshold:.4f} while model is below production sample threshold"
        )

    practical_context_ready = (
        not missing_critical_fields
        and pitcher_status in {"confirmed", "high_confidence_probable"}
        and starter_confidence_status == "known"
        and lineup_status == "confirmed"
        and bullpen_status == "available"
        and closer_status == "known"
    )

    effective_context_ready = False
    status = "context_tracking_only"

    if odds_status != "OK":
        status = "odds_blocked"
    elif model_source_norm != "ml":
        status = "model_blocked"
    elif selected_edge is None or selected_edge < min_moneyline_edge:
        status = "model_blocked"
    elif raw_context_ready:
        effective_context_ready = True
        status = "official_ready"
        reasons.append("strict official context is ready")
    elif practical_context_ready:
        effective_context_ready = True
        status = "practical_ready"
        reasons.append("effective betting context is practically ready")
    else:
        status = "context_tracking_only"

    score = 1.0
    unique_flags = set(risk_flags)

    if "odds_not_ok" in unique_flags:
        score -= 0.40
    if "non_ml_model" in unique_flags:
        score -= 0.30
    if "context_missing_fields" in unique_flags:
        score -= 0.30
    if "lineup_not_confirmed" in unique_flags:
        score -= 0.20
    if "bullpen_not_available" in unique_flags:
        score -= 0.15
    if "starter_confirmation_pending" in unique_flags:
        score -= 0.05
    if "closer_fatigue" in unique_flags:
        score -= 0.05
    if "closer_high_fatigue" in unique_flags:
        score -= 0.15
    if "edge_unavailable" in unique_flags or "edge_below_threshold" in unique_flags:
        score -= 0.20
    if "early_model_small_sample" in unique_flags:
        score -= 0.20
    if "large_anti_market_edge_early_model" in unique_flags:
        score -= 0.15

    score = round(_clamp_float(score), 4)
    if effective_context_ready and score < 0.75:
        effective_context_ready = False
        status = "risk_blocked"
        reasons.append("effective context was blocked by risk score below 0.75")

    if "large_anti_market_edge_early_model" in unique_flags:
        effective_context_ready = False
        status = "risk_blocked"
        reasons.append(
            "large anti-market edge blocked because model is still early-stage and recent CLV is negative"
        )

    if status == "official_ready" and score >= 0.90:
        stake_multiplier = 1.0
    elif status == "practical_ready" and score >= 0.85:
        stake_multiplier = 0.50
    elif status == "practical_ready" and score >= 0.75:
        stake_multiplier = 0.25
    else:
        stake_multiplier = 0.0

    if "large_anti_market_edge_early_model" in unique_flags:
        stake_multiplier = 0.0
        reasons.append(
            "stake multiplier forced to 0.00 because large anti-market edges are blocked during early model validation"
        )
    elif sample_count < production_sample_threshold:
        stake_multiplier = min(stake_multiplier, 0.10)
        reasons.append(
            "stake multiplier capped at 0.10 because model is still below production sample threshold"
        )

    return {
        "raw_context_ready_for_betting": raw_context_ready,
        "effective_context_ready_for_betting": bool(effective_context_ready),
        "betting_readiness_status": status,
        "betting_readiness_score": score,
        "betting_risk_flags": sorted(set(risk_flags)),
        "betting_readiness_reasons": reasons,
        "stake_multiplier": float(stake_multiplier),
        "moneyline_gate_status": gate_status,
        "selected_edge": (
            round(float(selected_edge), 4)
            if selected_edge is not None
            else None
        ),
        "model_training_sample_count": sample_count,
        "production_sample_threshold": production_sample_threshold,
        "early_model_guard_active": sample_count < production_sample_threshold,
        "large_edge_threshold": large_edge_threshold,
        "large_anti_market_edge_guard_active": (
            "large_anti_market_edge_early_model" in unique_flags
        ),
    }


def calculate_accuracy_safe_model_weight(
    *,
    sample_count: int,
    features: dict[str, Any],
    selected_edge_abs: float | None,
) -> tuple[float, list[str]]:
    """
    Return model weight for market blending.

    Lower model weight means stronger market anchoring.
    This is an accuracy/calibration safety layer, not a betting gate.
    """
    reasons: list[str] = []

    if sample_count < 120:
        model_weight = 0.55
        reasons.append("sample_count_below_120")
    elif sample_count < 300:
        model_weight = 0.65
        reasons.append("sample_count_below_300")
    elif sample_count < 500:
        model_weight = 0.75
        reasons.append("sample_count_below_500")
    else:
        model_weight = 0.85

    statcast_core_zero = (
        as_float(features.get("statcast_woba_diff"), 0.0) == 0.0
        and as_float(features.get("statcast_barrel_diff"), 0.0) == 0.0
        and as_float(features.get("statcast_hard_hit_diff"), 0.0) == 0.0
    )

    top3_zero = as_float(features.get("top3_woba_diff"), 0.0) == 0.0
    pitcher_advanced_zero = (
        as_float(features.get("sp_fip_diff"), 0.0) == 0.0
        and as_float(features.get("sp_csw_diff"), 0.0) == 0.0
    )

    if statcast_core_zero:
        model_weight = min(model_weight, 0.55)
        reasons.append("statcast_core_zero")

    if top3_zero:
        model_weight = min(model_weight, 0.55)
        reasons.append("top3_woba_zero")

    if pitcher_advanced_zero:
        model_weight = min(model_weight, 0.60)
        reasons.append("pitcher_advanced_zero")

    if selected_edge_abs is not None and selected_edge_abs >= 0.08:
        model_weight = min(model_weight, 0.50)
        reasons.append("large_edge_market_anchor")

    model_weight = float(np.clip(model_weight, 0.35, 0.90))
    return model_weight, reasons


def compute_market_no_vig_home_prob(
    home_odds: float | None,
    away_odds: float | None,
) -> float | None:
    """Return two-way no-vig home win probability from decimal odds."""
    if (
        home_odds is None
        or away_odds is None
        or home_odds <= 1
        or away_odds <= 1
    ):
        return None

    home_raw = 1.0 / home_odds
    away_raw = 1.0 / away_odds
    denominator = home_raw + away_raw

    if denominator <= 0:
        return None

    return home_raw / denominator

def kelly_criterion(
    win_prob: float | None,
    odds: float | None,
    fraction: float = 0.25,
) -> float:
    if win_prob is None or odds is None or odds <= 1:
        return 0.0

    edge_multiplier = odds - 1
    raw_fraction = win_prob - (1 - win_prob) / edge_multiplier
    return max(0.0, raw_fraction * fraction)


def build_recommendation_block_reason(
    *,
    recommendation_status: str,
    moneyline_gate_status: str,
    odds_are_usable: bool,
    odds_quality_status: str,
    suspicious_odds_reason: str,
    market_probability: float | None,
    model_edge_home: float | None,
    moneyline_selected_edge: float | None,
    min_moneyline_edge: float,
    model_source: str,
    model_error: str,
    daily_context_summary: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Explain why a game is or is not a Paper Bet.

    This helper is diagnostic only. It must not change recommendation logic,
    odds usage, model probabilities, Kelly sizing or snapshot behavior.
    """
    details: list[str] = []
    context = daily_context_summary or {}

    normalized_recommendation_status = str(
        recommendation_status or ""
    ).strip().upper()
    normalized_gate_status = str(moneyline_gate_status or "").strip().upper()
    normalized_odds_status = str(odds_quality_status or "UNAVAILABLE").strip().upper()

    if normalized_recommendation_status == "PAPER_BET":
        selected_edge = (
            moneyline_selected_edge
            if moneyline_selected_edge is not None
            else model_edge_home
        )
        if selected_edge is not None:
            details.append(
                "Selected edge "
                f"{selected_edge:.1%} passed threshold "
                f"{min_moneyline_edge:.1%}."
            )
        details.append("Odds quality passed.")
        details.append(f"Model source: {model_source or 'unknown'}.")
        if context:
            context_status = context.get("status")
            if context_status:
                details.append(f"Pregame context: {context_status}.")

            pitcher_status = context.get("pitcher_status")
            lineup_status = context.get("lineup_status")
            bullpen_status = context.get("bullpen_status")
            closer_status = context.get("closer_status")

            if pitcher_status or lineup_status or bullpen_status or closer_status:
                details.append(
                    "Context status: "
                    f"pitcher={pitcher_status or 'unknown'}, "
                    f"lineup={lineup_status or 'unknown'}, "
                    f"bullpen={bullpen_status or 'unknown'}, "
                    f"closer={closer_status or 'unknown'}."
                )

            lineup_projection_status = context.get("lineup_projection_status")
            if lineup_projection_status:
                details.append(
                    f"Lineup projection: {lineup_projection_status}."
                )

            home_starter_status = context.get("home_starter_status")
            away_starter_status = context.get("away_starter_status")
            home_starter_status = context.get("home_starter_status")
            away_starter_status = context.get("away_starter_status")
            if home_starter_status or away_starter_status:
                details.append(
                    "Starter confidence: "
                    f"home={home_starter_status or 'unknown'}, "
                    f"away={away_starter_status or 'unknown'}."
                )

            reason = context.get("context_not_ready_reason")
            if reason:
                details.append(str(reason))

        return "Paper bet edge passed", details

    if not odds_are_usable:
        details.append(f"Odds quality status: {normalized_odds_status}.")
        if suspicious_odds_reason:
            details.append(str(suspicious_odds_reason))
        else:
            details.append("No usable two-way market odds were available.")
        return "Odds unavailable or failed integrity checks", details

    if market_probability is None:
        details.append("No valid two-way no-vig market probability was available.")
        details.append(f"Odds quality status: {normalized_odds_status}.")
        return "Market probability unavailable", details

    if model_edge_home is None:
        details.append("Model edge could not be calculated.")
        details.append(f"Model source: {model_source or 'unknown'}.")
        if model_error:
            details.append(f"Model note: {model_error}")
        return "Model edge unavailable", details

    home_edge = float(model_edge_home)
    away_edge = -home_edge
    best_edge = max(home_edge, away_edge)

    if best_edge < min_moneyline_edge:
        details.append(
            "Best edge "
            f"{best_edge:.1%} is below required "
            f"{min_moneyline_edge:.1%}."
        )
        details.append(f"Home edge: {home_edge:.1%}; away edge: {away_edge:.1%}.")
        details.append(f"Gate status: {normalized_gate_status or 'UNKNOWN'}.")
        details.append(f"Model source: {model_source or 'unknown'}.")
        if model_error:
            details.append(f"Model note: {model_error}")
        return "Model edge below threshold", details

    if context and context.get("context_ready_for_betting") is False:
        reason = context.get("context_not_ready_reason")
        if reason:
            details.append(str(reason))

        missing = context.get("missing_critical_fields") or []
        if isinstance(missing, list) and missing:
            details.append(
                "Missing context: "
                + ", ".join(str(item) for item in missing[:5])
            )

        pitcher_status = context.get("pitcher_status")
        lineup_status = context.get("lineup_status")
        bullpen_status = context.get("bullpen_status")
        closer_status = context.get("closer_status")
        if pitcher_status or lineup_status or bullpen_status or closer_status:
            details.append(
                "Context status: "
                f"pitcher={pitcher_status or 'unknown'}, "
                f"lineup={lineup_status or 'unknown'}, "
                f"bullpen={bullpen_status or 'unknown'}, "
                f"closer={closer_status or 'unknown'}."
            )

        
        lineup_projection_status = context.get("lineup_projection_status")
        if lineup_projection_status:
            details.append(
                f"Lineup projection: {lineup_projection_status}."
            )

        home_starter_status = context.get("home_starter_status")
        away_starter_status = context.get("away_starter_status")
        if home_starter_status or away_starter_status:
            details.append(
                "Starter confidence: "
                f"home={home_starter_status or 'unknown'}, "
                f"away={away_starter_status or 'unknown'}."
            )

        return "Pregame context not ready", details

    details.append(f"Gate status: {normalized_gate_status or 'UNKNOWN'}.")
    details.append(f"Model source: {model_source or 'unknown'}.")
    if model_error:
        details.append(f"Model note: {model_error}")

    return "Tracking only", details


def get_season_phase_adjustment(date_str: str, pred_prob: float) -> float:
    month = datetime.strptime(date_str, "%Y-%m-%d").month
    adjustment = 0.0

    if month in (4, 5):
        adjustment = -0.005
    elif month == 9:
        if pred_prob > 0.8:
            adjustment = -0.123
        elif pred_prob > 0.7:
            adjustment = -0.088
        adjustment -= 0.025
    elif month == 10 and pred_prob > 0.7:
        adjustment = -0.10

    return adjustment


def load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        print(f"Unable to read {path}: {exc}")
        return {}


def load_historical_frames(
    errors: list[str],
) -> tuple[pd.DataFrame | None, int]:
    parquet_frames: list[pd.DataFrame] = []

    if HISTORICAL_DIR.exists():
        for file_path in sorted(HISTORICAL_DIR.glob("*.parquet")):
            try:
                frame = pd.read_parquet(file_path).dropna(how="all")
                if not frame.empty:
                    parquet_frames.append(frame)
            except Exception as exc:
                errors.append(f"Unable to read historical parquet {file_path}: {exc}")

    historical_df = (
        pd.concat(parquet_frames, ignore_index=True)
        if parquet_frames
        else None
    )

    historical_count = 0
    if HISTORY_FILE.exists():
        try:
            history = pd.read_csv(HISTORY_FILE)
            if "home_win" in history.columns:
                historical_count = int(history["home_win"].notna().sum())
        except Exception as exc:
            errors.append(f"Unable to read historical predictions: {exc}")

    return historical_df, historical_count


def schedule_diagnostics(
    schedule_df: pd.DataFrame,
    errors: list[str],
) -> tuple[bool, int | None]:
    critical_tokens = (
        "mlb_statsapi fetch error",
        "mlb_statsapi module not loaded",
        "schedule fetch failed",
    )
    lower_errors = [str(message).lower() for message in errors]

    schedule_failed = any(
        token in message
        for token in critical_tokens
        for message in lower_errors
    )

    if schedule_failed:
        return False, None

    return True, int(len(schedule_df))


def prepare_team_frame(raw_rows: Any) -> pd.DataFrame:
    frame = pd.DataFrame(raw_rows or [])

    required_defaults: dict[str, float] = {
        "wins": 0.0,
        "losses": 0.0,
        "runs_scored": 400.0,
        "runs_allowed": 400.0,
        "home_k_pct": 0.20,
        "home_bb_pct": 0.08,
        "away_k_pct": 0.20,
        "away_bb_pct": 0.08,
    }

    if "name" not in frame.columns:
        frame["name"] = pd.Series(dtype="object")

    frame["name"] = frame["name"].map(normalize_team)

    for column, default in required_defaults.items():
        if column not in frame.columns:
            frame[column] = default
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).fillna(default)

    games = frame["wins"] + frame["losses"]
    frame["win_pct"] = np.where(games > 0, frame["wins"] / games, 0.5)

    return frame


def load_ml_model() -> tuple[Any | None, list[str] | None, int, str]:
    """Load only a sufficiently trained model for the active clean pipeline."""
    if not MODEL_FILE.exists():
        return None, None, 0, "Model artifact does not exist."

    required_pipeline_version = getattr(
        config,
        "PIPELINE_VERSION",
        "baseline_v2_clean",
    )
    minimum_clean_samples = int(
        getattr(config, "MIN_CLEAN_TRAIN_SAMPLES", 300)
    )

    try:
        import joblib

        artifact = joblib.load(MODEL_FILE)
        if not (
            isinstance(artifact, dict)
            and "model" in artifact
            and "features" in artifact
        ):
            return None, None, 0, "Legacy model artifact is unsupported."

        artifact_pipeline_version = artifact.get("pipeline_version")
        training_sample_count = as_int(
            artifact.get("training_sample_count"),
            0,
        )

        if artifact_pipeline_version != required_pipeline_version:
            found_version = artifact_pipeline_version or "missing"
            return (
                None,
                None,
                training_sample_count,
                (
                    "Model artifact pipeline version mismatch: "
                    f"expected {required_pipeline_version}, "
                    f"found {found_version}."
                ),
            )

        if training_sample_count < minimum_clean_samples:
            return (
                None,
                None,
                training_sample_count,
                (
                    "Clean model artifact has insufficient training samples: "
                    f"{training_sample_count} < {minimum_clean_samples}."
                ),
            )

        print(
            "Loaded ML model artifact for pipeline "
            f"{required_pipeline_version} "
            f"with samples={training_sample_count}."
        )
        return (
            artifact["model"],
            list(artifact["features"]),
            training_sample_count,
            "",
        )

    except Exception as exc:
        return None, None, 0, str(exc)


def load_nrfi_model() -> tuple[Any | None, bool]:
    if not getattr(config, "NRFI_USE_ML", False) or NRFIModel is None:
        return None, False

    model_path = Path("models/nrf_model.pkl")
    if not model_path.exists():
        print("NRFI model artifact does not exist; using fallback behavior.")
        return None, False

    try:
        nrfi_model = NRFIModel(str(model_path))
        nrfi_model.load()
        print("Loaded NRFI model.")
        return nrfi_model, True
    except Exception as exc:
        print(f"Unable to load NRFI model: {exc}")
        return None, False


def build_schedule_frame(raw_rows: Any) -> pd.DataFrame:
    """Normalize scheduled games into the columns used by the predictor."""
    frame = pd.DataFrame(raw_rows or [])

    if frame.empty:
        return frame

    required_columns = ("game_id", "home_team", "away_team")
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Schedule data is missing columns: {missing_columns}")

    frame = frame.copy()
    frame["home_team"] = frame["home_team"].map(normalize_team)
    frame["away_team"] = frame["away_team"].map(normalize_team)

    if "game_date" not in frame.columns:
        frame["game_date"] = datetime.now().strftime("%Y-%m-%d")

    if "game_time" not in frame.columns:
        frame["game_time"] = ""

    if "start_time" not in frame.columns:
        frame["start_time"] = ""

    if "game_status" not in frame.columns:
        if "status" in frame.columns:
            frame["game_status"] = frame["status"]
        else:
            frame["game_status"] = ""

    if "venue" not in frame.columns:
        frame["venue"] = ""

    return frame


def build_pitcher_lookup(raw_rows: Any) -> dict[Any, pd.Series]:
    """Create a game_id to pitcher-data lookup."""
    frame = pd.DataFrame(raw_rows or [])
    if frame.empty or "game_id" not in frame.columns:
        return {}

    lookup: dict[Any, pd.Series] = {}
    for _, row in frame.iterrows():
        game_id = row.get("game_id")
        lookup[game_id] = row
        lookup[str(game_id)] = row

    return lookup


def get_team_row(team_frame: pd.DataFrame, team: str) -> pd.Series | None:
    if team_frame.empty or "name" not in team_frame.columns:
        return None

    rows = team_frame[team_frame["name"] == team]
    if rows.empty:
        return None

    return rows.iloc[0]


def calculate_team_winrate_difference(
    team_frame: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> float:
    home_row = get_team_row(team_frame, home_team)
    away_row = get_team_row(team_frame, away_team)

    home_pct = (
        as_float(home_row.get("win_pct"), 0.5)
        if home_row is not None
        else 0.5
    )
    away_pct = (
        as_float(away_row.get("win_pct"), 0.5)
        if away_row is not None
        else 0.5
    )

    return home_pct - away_pct


def calculate_dynamic_pythag_difference(
    team_frame: pd.DataFrame,
    home_team: str,
    away_team: str,
    exponent: float = 2.0,
) -> float:
    def team_pythag(team: str) -> float:
        row = get_team_row(team_frame, team)
        if row is None:
            return 0.5

        scored = max(as_float(row.get("runs_scored"), 400.0), 1.0)
        allowed = max(as_float(row.get("runs_allowed"), 400.0), 1.0)
        scored_power = scored ** exponent
        allowed_power = allowed ** exponent
        return scored_power / (scored_power + allowed_power)

    return team_pythag(home_team) - team_pythag(away_team)


def calculate_timezone_difference(home_team: str, away_team: str) -> float:
    home_zone = TEAM_TIMEZONES.get(home_team, "Eastern")
    away_zone = TEAM_TIMEZONES.get(away_team, "Eastern")
    return float(
        TIMEZONE_OFFSETS.get(home_zone, 0)
        - TIMEZONE_OFFSETS.get(away_zone, 0)
    )


def detect_day_game(game: pd.Series) -> float:
    value = str(game.get("game_time", "")).lower()
    if any(
        token in value
        for token in ("am", "12:", "13:", "1:", "14:", "2:", "15:", "3:")
    ):
        return 1.0
    return as_float(game.get("is_day_game"), 0.0)


def load_last_game_data() -> dict[str, Any]:
    return load_json_dict(LAST_GAME_FILE)


def calculate_back_to_back_difference(
    home_team: str,
    away_team: str,
    date_str: str,
    last_game_data: dict[str, Any],
) -> float:
    try:
        current_date = pd.to_datetime(date_str)

        def is_back_to_back(team: str) -> float:
            last_date = last_game_data.get(team)
            if not last_date:
                return 0.0
            previous_date = pd.to_datetime(last_date)
            return 1.0 if (current_date - previous_date).days == 1 else 0.0

        return is_back_to_back(home_team) - is_back_to_back(away_team)
    except Exception:
        return 0.0


def extract_odds_values(
    odds_data: Any,
    game_id: Any,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    str,
    str,
    str,
    list[dict[str, Any]],
]:
    """Return audited odds values and quality metadata for one scheduled game."""
    if isinstance(odds_data, pd.DataFrame):
        frame = odds_data.copy()
    else:
        frame = pd.DataFrame(odds_data or [])

    unavailable_result = (
        None,
        None,
        None,
        None,
        "UNAVAILABLE",
        "No audited odds market matched this scheduled game.",
        "unknown",
        [],
    )

    if frame.empty or "game_id" not in frame.columns:
        return unavailable_result

    matched = frame[frame["game_id"].astype(str) == str(game_id)]
    if matched.empty:
        return unavailable_result

    row = matched.iloc[0]

    home_odds = None
    away_odds = None
    total_line = None
    spread_line = None

    for column in (
        "home_odds",
        "home_moneyline_odds",
        "home_decimal_odds",
        "moneyline_home_odds",
    ):
        if column in row.index:
            value = as_float(row.get(column), 0.0)
            if value > 1:
                home_odds = value
                break

    for column in (
        "away_odds",
        "away_moneyline_odds",
        "away_decimal_odds",
        "moneyline_away_odds",
    ):
        if column in row.index:
            value = as_float(row.get(column), 0.0)
            if value > 1:
                away_odds = value
                break

    for column in ("total_line", "over_under", "total"):
        if column in row.index and pd.notna(row.get(column)):
            parsed_total = as_float(row.get(column), 0.0)
            total_line = parsed_total if parsed_total > 0 else None
            break

    for column in ("spread_line", "home_spread", "spread"):
        if column in row.index and pd.notna(row.get(column)):
            spread_line = as_float(row.get(column), 0.0)
            break

    odds_quality_status = str(
        row.get("odds_quality_status", "") or ""
    ).strip().upper()
    if odds_quality_status not in {"OK", "SUSPICIOUS", "UNAVAILABLE"}:
        odds_quality_status = (
            "OK"
            if home_odds is not None and away_odds is not None
            else "UNAVAILABLE"
        )

    suspicious_odds_reason = str(
        row.get("suspicious_odds_reason", "") or ""
    ).strip()
    if odds_quality_status != "OK" and not suspicious_odds_reason:
        suspicious_odds_reason = "Odds market did not pass integrity validation."

    odds_source = str(
        row.get("odds_source", "unknown") or "unknown"
    ).strip()

    bookmaker_quotes = row.get("bookmaker_quotes", [])
    if isinstance(bookmaker_quotes, str):
        try:
            bookmaker_quotes = json.loads(bookmaker_quotes)
        except (TypeError, ValueError):
            bookmaker_quotes = []
    if not isinstance(bookmaker_quotes, list):
        bookmaker_quotes = []

    return (
        home_odds,
        away_odds,
        total_line,
        spread_line,
        odds_quality_status,
        suspicious_odds_reason,
        odds_source,
        bookmaker_quotes,
    )
def calculate_elo_features(
    home_team: str,
    away_team: str,
    errors: list[str],
) -> tuple[float, float, str]:
    """Return rating difference, neutral rating difference and rating source."""
    default_diff = 0.0

    if getattr(config, "RATINGS_ENGINE", "elo") == "glicko2":
        if load_glicko2_league is None:
            errors.append("Glicko2 is enabled but loader is unavailable.")
            return default_diff, default_diff, "unavailable"

        try:
            league = load_glicko2_league()
            if home_team not in league.teams or away_team not in league.teams:
                errors.append(
                    f"Glicko2 teams missing for matchup: {home_team} vs {away_team}"
                )
                return default_diff, default_diff, "glicko2_missing_team"

            rating_diff = float(
                league.teams[home_team].rating - league.teams[away_team].rating
            )
            return rating_diff, rating_diff, "glicko2"
        except Exception as exc:
            errors.append(f"Glicko2 rating lookup failed: {exc}")
            return default_diff, default_diff, "glicko2_error"

    if MLBElosystem is None:
        errors.append("Elo system is unavailable.")
        return default_diff, default_diff, "unavailable"

    try:
        elo_system = MLBElosystem()
        home_rating = as_float(elo_system.get_rating(home_team), 1500.0)
        away_rating = as_float(elo_system.get_rating(away_team), 1500.0)
        home_advantage = as_float(
            getattr(elo_system, "home_adv", 24.0),
            24.0,
        )

        feature_diff = home_rating - away_rating + home_advantage
        neutral_diff = feature_diff - home_advantage
        return feature_diff, neutral_diff, "elo"
    except Exception as exc:
        errors.append(f"Elo rating lookup failed: {exc}")
        return default_diff, default_diff, "elo_error"


def calculate_statcast_features(
    statcast_frame: pd.DataFrame,
    home_team: str,
    away_team: str,
    errors: list[str],
) -> dict[str, float]:
    features = {
        "statcast_launch_speed_diff": 0.0,
        "statcast_barrel_diff": 0.0,
        "statcast_hard_hit_diff": 0.0,
        "statcast_woba_diff": 0.0,
        "pitch_movement_diff": 0.0,
        "avg_bat_speed_diff": 0.0,
        "sprint_speed_diff": 0.0,
        "swing_miss_diff": 0.0,
        "csw_diff": 0.0,
        "barrel_pa_diff": 0.0,
        "hardhit_pa_diff": 0.0,
        "barrel_bb_pct_diff": 0.0,
    }

    if statcast_frame.empty:
        return features

    if "batter_team" not in statcast_frame.columns:
        errors.append("Statcast rows are unavailable for team aggregation.")
        return features

    frame = statcast_frame.copy()
    frame["batter_team"] = frame["batter_team"].map(normalize_team)

    home_rows = frame[frame["batter_team"] == home_team]
    away_rows = frame[frame["batter_team"] == away_team]

    if home_rows.empty or away_rows.empty:
        return features

    column_map = {
        "statcast_launch_speed_diff": "launch_speed",
        "statcast_barrel_diff": "barrel",
        "statcast_hard_hit_diff": "hard_hit",
        "statcast_woba_diff": "expected_woba",
        "avg_bat_speed_diff": "bat_speed",
        "sprint_speed_diff": "sprint_speed",
        "swing_miss_diff": "whiff",
        "csw_diff": "csw",
    }

    for feature_name, column_name in column_map.items():
        if column_name in frame.columns:
            home_mean = pd.to_numeric(
                home_rows[column_name],
                errors="coerce",
            ).mean()
            away_mean = pd.to_numeric(
                away_rows[column_name],
                errors="coerce",
            ).mean()
            if pd.notna(home_mean) and pd.notna(away_mean):
                features[feature_name] = float(home_mean - away_mean)

    if "pfx_x" in frame.columns and "pfx_z" in frame.columns:
        home_movement = (
            pd.to_numeric(home_rows["pfx_x"], errors="coerce").abs().mean()
            + pd.to_numeric(home_rows["pfx_z"], errors="coerce").abs().mean()
        )
        away_movement = (
            pd.to_numeric(away_rows["pfx_x"], errors="coerce").abs().mean()
            + pd.to_numeric(away_rows["pfx_z"], errors="coerce").abs().mean()
        )
        if pd.notna(home_movement) and pd.notna(away_movement):
            features["pitch_movement_diff"] = float(
                home_movement - away_movement
            )

    return features


def calculate_manual_nrfi(
    game: pd.Series,
    pitcher_data: pd.Series | None,
) -> tuple[float | None, str, str, str]:
    """Return NRFI probability, recommendation, source and fallback reason."""
    home_top3_woba = as_float(game.get("home_top3_avg_woba"), 0.320)
    away_top3_woba = as_float(game.get("away_top3_avg_woba"), 0.320)

    home_first_era = (
        pitcher_data.get("home_first_era")
        if pitcher_data is not None
        else None
    )
    away_first_era = (
        pitcher_data.get("away_first_era")
        if pitcher_data is not None
        else None
    )

    has_manual_inputs = (
        pitcher_data is not None
        and pd.notna(home_first_era)
        and pd.notna(away_first_era)
        and home_top3_woba != 0.320
        and away_top3_woba != 0.320
    )

    if not has_manual_inputs:
        return (
            None,
            "NO DATA",
            "unavailable",
            "Missing first-inning ERA or top-three wOBA data.",
        )

    home_first_era_value = as_float(home_first_era, 4.5)
    away_first_era_value = as_float(away_first_era, 4.5)
    average_first_era = (home_first_era_value + away_first_era_value) / 2
    top_three_factor = (home_top3_woba + away_top3_woba) / (2 * 0.320)

    nrfi_probability = 0.5 + (4.5 - average_first_era) * 0.08
    nrfi_probability = nrfi_probability / max(top_three_factor, 0.50)
    nrfi_probability = float(np.clip(nrfi_probability, 0.05, 0.95))

    recommendation = "NRFI" if nrfi_probability >= 0.55 else "NO BET"
    return nrfi_probability, recommendation, "manual", ""


def generate_predictions() -> dict[str, Any]:
    """Gather inputs, create daily predictions, and return the output report."""
    errors: list[str] = []
    predictions: list[dict[str, Any]] = []

    odds_history_captured_at = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    market_odds_history_summary: dict[str, Any] = {
        "status": (
            "completed"
            if append_game_market_snapshot is not None
            else "unavailable"
        ),
        "captured_at": odds_history_captured_at,
        "received": 0,
        "inserted": 0,
        "duplicates": 0,
        "stored_rows": 0,
        "errors": [],
    }

    (
        ml_model,
        model_features,
        clean_model_sample_count,
        model_load_error,
    ) = load_ml_model()
    if model_load_error:
        print(f"ML model unavailable: {model_load_error}")

    nrfi_model, nrfi_model_available = load_nrfi_model()

    try:
        gathered_data = UnifiedSportsModel().gather_all_data()
        if not isinstance(gathered_data, dict):
            raise TypeError("UnifiedSportsModel.gather_all_data() must return a dict.")
    except Exception as exc:
        errors.append(f"Data gathering failed: {exc}")
        gathered_data = {}

    raw_schedule = gathered_data.get("mlb_statsapi", [])
    try:
        schedule_frame = build_schedule_frame(raw_schedule)
    except Exception as exc:
        errors.append(f"Schedule data normalization failed: {exc}")
        schedule_frame = pd.DataFrame()

    schedule_fetch_ok, scheduled_game_count = schedule_diagnostics(
        schedule_frame,
        errors,
    )

    team_frame = prepare_team_frame(gathered_data.get("mlb_team_stats", []))
    pitcher_lookup = build_pitcher_lookup(gathered_data.get("pitcher_data", []))

    statcast_frame = pd.DataFrame(
        gathered_data.get("savant_statcast", []) or []
    )
    odds_data = gathered_data.get("odds", [])
    weather_frame = pd.DataFrame(gathered_data.get("weather", []) or [])
    injury_frame = pd.DataFrame(gathered_data.get("injuries", []) or [])
    catcher_frame = pd.DataFrame(gathered_data.get("catcher_data", []) or [])
    bullpen_frame = pd.DataFrame(gathered_data.get("bullpen_data", []) or [])

    historical_df, _ = load_historical_frames(errors)
    last_game_data = load_last_game_data()

    daily_context_by_game, daily_context_load_summary = (
        latest_daily_context_by_game(errors)
    )    
    savant_top3_by_game = latest_savant_top3_context_by_game(errors)
    weather_context_by_game = latest_weather_context_by_game(errors)
    pitcher_advanced_by_game = latest_context_csv_by_game(
        PITCHER_ADVANCED_CONTEXT_FILE,
        "pitcher_advanced_captured_at",
        errors,
        "pitcher advanced context",
    )
    context_bridge_by_game = latest_context_csv_by_game(
        CONTEXT_FEATURE_BRIDGE_FILE,
        "context_bridge_captured_at",
        errors,
        "context feature bridge",
    )
    team_form_by_game = latest_context_csv_by_game(
        TEAM_FORM_CONTEXT_FILE,
        "team_form_captured_at",
        errors,
        "team form context",
    )

    dynamic_pythag_exponent = 2.0
    if not team_frame.empty:
        run_scored_total = as_float(team_frame["runs_scored"].sum(), 0.0)
        run_allowed_total = as_float(team_frame["runs_allowed"].sum(), 0.0)
        if run_scored_total > 0 and run_allowed_total > 0:
            dynamic_pythag_exponent = 2.0

    bradley_terry_strengths: dict[str, float] = {}
    if get_bradley_terry_strengths is not None:
        try:
            values = get_bradley_terry_strengths()
            if isinstance(values, dict):
                bradley_terry_strengths = {
                    normalize_team(team): as_float(strength, 0.0)
                    for team, strength in values.items()
                }
        except Exception as exc:
            errors.append(f"Bradley-Terry feature failed: {exc}")

    if schedule_frame.empty:
        if schedule_fetch_ok:
            print("No scheduled MLB games found for this run.")
        else:
            errors.append("Schedule fetch failed and no predictions were generated.")

    for _, game in schedule_frame.iterrows():
        game_id = game.get("game_id")
        home_team = normalize_team(game.get("home_team"))
        away_team = normalize_team(game.get("away_team"))
        date_str = str(
            game.get("game_date")
            or datetime.now().strftime("%Y-%m-%d")
        )[:10]
        start_time = str(game.get("start_time") or "")
        game_status = str(
            game.get("game_status")
            or game.get("status")
            or ""
        )
        context_row = daily_context_by_game.get(str(game_id), {})
        savant_top3_row = savant_top3_by_game.get(str(game_id), {})
        weather_context_row = weather_context_by_game.get(str(game_id), {})
        weather_context_summary = build_weather_context_summary(weather_context_row)
        pitcher_advanced_row = pitcher_advanced_by_game.get(str(game_id), {})
        context_bridge_row = context_bridge_by_game.get(str(game_id), {})
        team_form_row = team_form_by_game.get(str(game_id), {})
        daily_context_summary = build_daily_context_summary(
            game_id,
            daily_context_by_game,
        )
        
        if not home_team or not away_team:
            errors.append(f"Skipping game {game_id}: missing team names.")
            continue

        features = {feature: 0.0 for feature in EXPECTED_FEATURES}

        feature_elo_diff, neutral_elo_diff, rating_source = calculate_elo_features(
            home_team,
            away_team,
            errors,
        )
        features["elo_diff"] = round(feature_elo_diff, 4)

        pitcher_data = pitcher_lookup.get(game_id)
        if pitcher_data is None:
            pitcher_data = pitcher_lookup.get(str(game_id))

        sp_era_diff = 0.0
        sp_fip_diff = 0.0
        sp_stuff_plus_diff = 0.0
        sp_csw_diff = 0.0
        home_sp_era = 4.5
        away_sp_era = 4.5

        if pitcher_data is not None:
            home_sp_era = as_float(pitcher_data.get("home_era"), 4.5)
            away_sp_era = as_float(pitcher_data.get("away_era"), 4.5)
            sp_era_diff = home_sp_era - away_sp_era

            home_fip = as_float(pitcher_data.get("home_fip"), 4.5)
            away_fip = as_float(pitcher_data.get("away_fip"), 4.5)
            sp_fip_diff = home_fip - away_fip

            home_stuff = as_float(pitcher_data.get("home_stuff_plus"), 100.0)
            away_stuff = as_float(pitcher_data.get("away_stuff_plus"), 100.0)
            sp_stuff_plus_diff = home_stuff - away_stuff

            home_csw = as_float(pitcher_data.get("home_csw_pct"), 0.28)
            away_csw = as_float(pitcher_data.get("away_csw_pct"), 0.28)
            sp_csw_diff = home_csw - away_csw

        features["sp_era_diff"] = round(sp_era_diff, 4)
        features["sp_fip_diff"] = round(sp_fip_diff, 4)
        features["sp_stuff_plus_diff"] = round(sp_stuff_plus_diff, 4)
        features["sp_csw_diff"] = round(sp_csw_diff, 4)

        features["winrate_diff"] = round(
            calculate_team_winrate_difference(team_frame, home_team, away_team),
            4,
        )
        features["dynamic_pythag_diff"] = round(
            calculate_dynamic_pythag_difference(
                team_frame,
                home_team,
                away_team,
                exponent=dynamic_pythag_exponent,
            ),
            4,
        )
        features["timezone_diff"] = calculate_timezone_difference(
            home_team,
            away_team,
        )
        features["is_day_game"] = detect_day_game(game)
        features["back2back_diff"] = calculate_back_to_back_difference(
            home_team,
            away_team,
            date_str,
            last_game_data,
        )

        if get_park_factor is not None:
            try:
                venue = str(
                    context_row.get("venue_name")
                    or game.get("venue")
                    or ""
                )
                features["dynamic_park_factor"] = as_float(
                    get_park_factor(venue),
                    1.0,
                )
            except Exception as exc:
                errors.append(f"Park factor failed for game {game_id}: {exc}")
                features["dynamic_park_factor"] = 1.0
        else:
            features["dynamic_park_factor"] = 1.0

        if calculate_lag_features is not None and historical_df is not None:
            try:
                lag_winrate_diff, lag_runs_diff = calculate_lag_features(
                    home_team,
                    away_team,
                    historical_df,
                    date_str,
                    days=30,
                )
                features["lag30_winrate_diff"] = as_float(
                    lag_winrate_diff,
                    0.0,
                )
                features["lag30_runs_diff"] = as_float(
                    lag_runs_diff,
                    0.0,
                )
            except Exception as exc:
                errors.append(f"Lag feature failed for game {game_id}: {exc}")

        if calculate_catcher_effect is not None and not catcher_frame.empty:
            try:
                catcher_result = calculate_catcher_effect(
                    catcher_frame,
                    home_team,
                    away_team,
                )
                if isinstance(catcher_result, dict):
                    features["catcher_era_diff"] = as_float(
                        catcher_result.get("catcher_era_diff"),
                        0.0,
                    )
                    features["cs_diff"] = as_float(
                        catcher_result.get("cs_diff"),
                        0.0,
                    )
                elif (
                    isinstance(catcher_result, (tuple, list))
                    and len(catcher_result) >= 2
                ):
                    features["catcher_era_diff"] = as_float(
                        catcher_result[0],
                        0.0,
                    )
                    features["cs_diff"] = as_float(
                        catcher_result[1],
                        0.0,
                    )
            except Exception as exc:
                errors.append(f"Catcher feature failed for game {game_id}: {exc}")

        if (
            calculate_bullpen_availability is not None
            and not bullpen_frame.empty
        ):
            try:
                bullpen_result = calculate_bullpen_availability(
                    bullpen_frame,
                    home_team,
                    away_team,
                )
                if isinstance(bullpen_result, dict):
                    features["bullpen_ip_diff"] = as_float(
                        bullpen_result.get("bullpen_ip_diff"),
                        0.0,
                    )
                    features["bullpen_availability_diff"] = as_float(
                        bullpen_result.get("bullpen_availability_diff"),
                        0.0,
                    )
                elif (
                    isinstance(bullpen_result, (tuple, list))
                    and len(bullpen_result) >= 2
                ):
                    features["bullpen_ip_diff"] = as_float(
                        bullpen_result[0],
                        0.0,
                    )
                    features["bullpen_availability_diff"] = as_float(
                        bullpen_result[1],
                        0.0,
                    )
            except Exception as exc:
                errors.append(f"Bullpen feature failed for game {game_id}: {exc}")

        statcast_features = calculate_statcast_features(
            statcast_frame,
            home_team,
            away_team,
            errors,
        )
        features.update(statcast_features)
        (
            home_odds,
            away_odds,
            total_line,
            spread_line,
            odds_quality_status,
            suspicious_odds_reason,
            odds_source,
            bookmaker_quotes,
        ) = extract_odds_values(
            odds_data,
            game_id,
        )

        odds_are_usable = odds_quality_status == "OK"
        market_probability = (
            compute_market_no_vig_home_prob(home_odds, away_odds)
            if odds_are_usable
            else None
        )
        if append_game_market_snapshot is not None and bookmaker_quotes:
            try:
                odds_history_result = append_game_market_snapshot(
                    game_id=game_id,
                    game_date=date_str,
                    start_time=start_time,
                    captured_at=odds_history_captured_at,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker_quotes=bookmaker_quotes,
                    odds_quality_status=odds_quality_status,
                    suspicious_odds_reason=suspicious_odds_reason,
                    starting_pitcher_confirmed=(
                        daily_context_summary.get("pitcher_status") == "confirmed"
                    ),
                    lineup_confirmed=(
                        daily_context_summary.get("lineup_status") == "confirmed"
                    ),
                )

                for summary_key in ("received", "inserted", "duplicates"):
                    market_odds_history_summary[summary_key] += int(
                        odds_history_result.get(summary_key, 0)
                    )

                market_odds_history_summary["stored_rows"] = max(
                    int(market_odds_history_summary.get("stored_rows", 0)),
                    int(odds_history_result.get("stored_rows", 0)),
                )

                odds_history_errors = odds_history_result.get("errors", [])
                if odds_history_errors:
                    market_odds_history_summary["status"] = "partial_failure"
                    market_odds_history_summary["errors"].extend(
                        [
                            f"Game {game_id}: {message}"
                            for message in odds_history_errors
                        ]
                    )

            except Exception as exc:
                market_odds_history_summary["status"] = "partial_failure"
                market_odds_history_summary["errors"].append(
                    f"Game {game_id}: {exc}"
                )
        
        if not odds_are_usable:
            errors.append(
                (
                    f"Tracking only for {away_team} at {home_team}: "
                    f"odds_quality_status={odds_quality_status}; "
                    f"reason={suspicious_odds_reason}"
                )
            )

        if getattr(config, "ODDS_USE_CURVE_FEATURES", False):
            features["odds_change"] = as_float(game.get("odds_change"), 0.0)

        if not weather_frame.empty:
            try:
                if "game_id" in weather_frame.columns:
                    rows = weather_frame[
                        weather_frame["game_id"].astype(str) == str(game_id)
                    ]
                    weather = (
                        rows.iloc[0]
                        if not rows.empty
                        else weather_frame.iloc[0]
                    )
                else:
                    weather = weather_frame.iloc[0]

                features["wind_effect"] = as_float(
                    weather.get("wind_effect"),
                    0.0,
                )
                features["temp_effect"] = as_float(
                    weather.get("temp_effect"),
                    0.0,
                )
                features["precip_effect"] = as_float(
                    weather.get("precip_effect"),
                    0.0,
                )
            except Exception as exc:
                errors.append(f"Weather feature failed for game {game_id}: {exc}")

        context_weather_features = calculate_context_weather_features(context_row)

        if features.get("wind_effect", 0.0) == 0.0:
            features["wind_effect"] = context_weather_features["wind_effect"]

        if features.get("temp_effect", 0.0) == 0.0:
            features["temp_effect"] = context_weather_features["temp_effect"]

        if features.get("precip_effect", 0.0) == 0.0:
            features["precip_effect"] = context_weather_features["precip_effect"]

        if weather_context_row:
            for feature_name in ("temp_effect", "wind_effect", "precip_effect"):
                if feature_name in weather_context_row:
                    raw_weather_value = weather_context_row.get(feature_name)
                    raw_weather_text = str(raw_weather_value).strip().lower()
                    if raw_weather_text not in {"", "nan", "none", "null"}:
                        features[feature_name] = round(
                            as_float(raw_weather_value, 0.0),
                            4,
                        )
                        
        if not injury_frame.empty:
            try:
                if "team" in injury_frame.columns:
                    injury_frame_normalized = injury_frame.copy()
                    injury_frame_normalized["team"] = (
                        injury_frame_normalized["team"].map(normalize_team)
                    )
                    home_injuries = injury_frame_normalized[
                        injury_frame_normalized["team"] == home_team
                    ]
                    away_injuries = injury_frame_normalized[
                        injury_frame_normalized["team"] == away_team
                    ]
                    home_value = (
                        pd.to_numeric(
                            home_injuries.get("impact", pd.Series(dtype=float)),
                            errors="coerce",
                        )
                        .fillna(0.0)
                        .sum()
                    )
                    away_value = (
                        pd.to_numeric(
                            away_injuries.get("impact", pd.Series(dtype=float)),
                            errors="coerce",
                        )
                        .fillna(0.0)
                        .sum()
                    )
                    features["injury_diff"] = float(home_value - away_value)
            except Exception as exc:
                errors.append(f"Injury feature failed for game {game_id}: {exc}")

        if get_elo_momentum is not None:
            try:
                momentum_result = get_elo_momentum(home_team, away_team)
                if isinstance(momentum_result, dict):
                    features["elo_momentum_7d"] = as_float(
                        momentum_result.get("elo_momentum_7d"),
                        0.0,
                    )
                    features["elo_momentum_30d"] = as_float(
                        momentum_result.get("elo_momentum_30d"),
                        0.0,
                    )
            except Exception as exc:
                errors.append(f"Elo momentum failed for game {game_id}: {exc}")

        if (
            getattr(config, "FEATURE_USE_PITCH_MATCHUP", False)
            and get_pitch_type_matchup_score is not None
        ):
            try:
                features["pitch_type_matchup_score"] = as_float(
                    get_pitch_type_matchup_score(home_team, away_team),
                    0.0,
                )
            except Exception as exc:
                errors.append(f"Pitch matchup failed for game {game_id}: {exc}")

        home_top3_count = as_int(
            savant_top3_row.get("home_top3_savant_available_count"),
            0,
        )
        away_top3_count = as_int(
            savant_top3_row.get("away_top3_savant_available_count"),
            0,
        )

        savant_top3_available = home_top3_count > 0 and away_top3_count > 0

        if savant_top3_available:
            features["top3_woba_diff"] = round(
                as_float(savant_top3_row.get("top3_woba_diff"), 0.0),
                4,
            )

            if features.get("statcast_woba_diff", 0.0) == 0.0:
                features["statcast_woba_diff"] = round(
                    as_float(savant_top3_row.get("top3_xwoba_diff"), 0.0),
                    4,
                )

            if features.get("statcast_hard_hit_diff", 0.0) == 0.0:
                features["statcast_hard_hit_diff"] = round(
                    as_float(savant_top3_row.get("top3_hard_hit_rate_diff"), 0.0),
                    4,
                )

            if features.get("statcast_barrel_diff", 0.0) == 0.0:
                features["statcast_barrel_diff"] = round(
                    as_float(savant_top3_row.get("top3_barrel_rate_diff"), 0.0),
                    4,
                )

            if features.get("statcast_launch_speed_diff", 0.0) == 0.0:
                features["statcast_launch_speed_diff"] = round(
                    as_float(savant_top3_row.get("top3_avg_launch_speed_diff"), 0.0),
                    4,
                )

            if features.get("avg_bat_speed_diff", 0.0) == 0.0:
                features["avg_bat_speed_diff"] = round(
                    as_float(savant_top3_row.get("top3_avg_launch_speed_diff"), 0.0),
                    4,
                )

            if features.get("barrel_pa_diff", 0.0) == 0.0:
                features["barrel_pa_diff"] = round(
                    as_float(savant_top3_row.get("top3_barrel_rate_diff"), 0.0),
                    4,
                )

            if features.get("hardhit_pa_diff", 0.0) == 0.0:
                features["hardhit_pa_diff"] = round(
                    as_float(savant_top3_row.get("top3_hard_hit_rate_diff"), 0.0),
                    4,
                )
        else:
            features["top3_woba_diff"] = 0.0

        features["bt_strength_diff"] = (
            bradley_terry_strengths.get(home_team, 0.0)
            - bradley_terry_strengths.get(away_team, 0.0)
        )

        home_row = get_team_row(team_frame, home_team)
        away_row = get_team_row(team_frame, away_team)
        if home_row is not None and away_row is not None:
            features["k_pct_diff"] = (
                as_float(home_row.get("home_k_pct"), 0.20)
                - as_float(away_row.get("away_k_pct"), 0.20)
            )
            features["bb_pct_diff"] = (
                as_float(home_row.get("home_bb_pct"), 0.08)
                - as_float(away_row.get("away_bb_pct"), 0.08)
            )

        if pitcher_advanced_row:
            for feature_name in (
                "sp_fip_diff",
                "sp_csw_diff",
                "sp_stuff_plus_diff",
                "k_pct_diff",
                "bb_pct_diff",
            ):
                raw_value = pitcher_advanced_row.get(feature_name)
                raw_text = str(raw_value).strip().lower()
                if raw_text not in {"", "nan", "none", "null"}:
                    features[feature_name] = round(
                        as_float(raw_value, 0.0),
                        4,
                    )

        if context_bridge_row:
            for feature_name in (
                "bullpen_ip_diff",
                "bullpen_availability_diff",
            ):
                raw_value = context_bridge_row.get(feature_name)
                raw_text = str(raw_value).strip().lower()
                if raw_text not in {"", "nan", "none", "null"}:
                    features[feature_name] = round(
                        as_float(raw_value, 0.0),
                        4,
                    )
                    
        if team_form_row:
            team_form_status = str(
                team_form_row.get("team_form_source_status", "")
            ).strip().lower()

            if team_form_status == "ok":
                team_form_bounds = {
                    "lag30_winrate_diff": (-0.75, 0.75),
                    "lag30_runs_diff": (-5.0, 5.0),
                    "rest_diff": (-3.0, 3.0),
                    "log5_prob": (0.25, 0.75),
                    "elo_momentum_7d": (-1.0, 1.0),
                    "elo_momentum_30d": (-1.0, 1.0),
                }

                for feature_name, bounds in team_form_bounds.items():
                    raw_value = team_form_row.get(feature_name)
                    raw_text = str(raw_value).strip().lower()
                    if raw_text in {"", "nan", "none", "null"}:
                        continue

                    value = as_float(raw_value, None)
                    if value is None:
                        continue

                    lower_bound, upper_bound = bounds
                    value = max(lower_bound, min(upper_bound, value))
                    features[feature_name] = round(value, 4)
                    
        if calculate_pitcher_ratings is not None and pitcher_data is not None:
            try:
                rating_result = calculate_pitcher_ratings(pitcher_data)
                if isinstance(rating_result, dict):
                    features["pitcher_rating_diff"] = as_float(
                        rating_result.get("pitcher_rating_diff"),
                        0.0,
                    )
            except Exception as exc:
                errors.append(f"Pitcher rating failed for game {game_id}: {exc}")

        manual_elo_probability = 1 / (
            1 + 10 ** (-neutral_elo_diff / 400.0)
        )
        clipped_sp_era = float(np.clip(sp_era_diff, -2.0, 2.0))
        starter_adjustment = -0.03 * clipped_sp_era
        manual_prediction = float(
            np.clip(manual_elo_probability + starter_adjustment, 0.05, 0.95)
        )

        no_odds_prediction = manual_prediction
        model_source = "manual"
        model_error = model_load_error

        if ml_model is not None and model_features:
            try:
                model_array = np.array(
                    [[features.get(feature, 0.0) for feature in model_features]],
                    dtype=float,
                )
                ml_prediction = float(ml_model.predict_proba(model_array)[0, 1])
                ml_weight = min(0.50, clean_model_sample_count / 1000.0)
                predicted_home_win = (
                    (1 - ml_weight) * manual_prediction
                    + ml_weight * ml_prediction
                )
                model_source = "ml"
                model_error = ""
            except Exception as exc:
                predicted_home_win = manual_prediction
                model_error = f"ML prediction failed: {exc}"
                errors.append(f"ML prediction failed for game {game_id}: {exc}")
        else:
            predicted_home_win = manual_prediction

        predicted_home_win += get_season_phase_adjustment(
            date_str,
            predicted_home_win,
        )

        predicted_home_win = float(np.clip(predicted_home_win, 0.05, 0.95))
        premarket_model_home_prob = predicted_home_win

        model_edge_home = (
            premarket_model_home_prob - market_probability
            if market_probability is not None
            else None
        )

        market_adjustment_applied = False
        accuracy_safe_model_weight = 1.0
        accuracy_safe_market_weight = 0.0
        accuracy_safe_blend_reasons: list[str] = []

        if market_probability is not None:
            selected_edge_abs_for_blend = (
                abs(model_edge_home)
                if model_edge_home is not None
                else None
            )

            accuracy_safe_model_weight, accuracy_safe_blend_reasons = (
                calculate_accuracy_safe_model_weight(
                    sample_count=clean_model_sample_count,
                    features=features,
                    selected_edge_abs=selected_edge_abs_for_blend,
                )
            )
            accuracy_safe_market_weight = 1.0 - accuracy_safe_model_weight

            predicted_home_win = (
                premarket_model_home_prob * accuracy_safe_model_weight
                + market_probability * accuracy_safe_market_weight
            )
            market_adjustment_applied = True
            
        predicted_home_win = float(np.clip(predicted_home_win, 0.05, 0.95))
        displayed_home_win_pct = predicted_home_win

        over_probability = None
        home_cover_probability = None
        away_cover_probability = None

        simulation_total_line = total_line if odds_are_usable else None
        simulation_spread_line = spread_line if odds_are_usable else None

        if MonteCarloSimulator is not None:
            try:
                simulator = MonteCarloSimulator()
                simulation_result = simulator.simulate(
                    home_team=home_team,
                    away_team=away_team,
                    home_sp_era=home_sp_era,
                    away_sp_era=away_sp_era,
                    park_factor=features["dynamic_park_factor"],
                    total_line=simulation_total_line,
                    spread_line=simulation_spread_line,
                )
                if isinstance(simulation_result, dict):
                    over_probability = simulation_result.get("over_prob")
                    home_cover_probability = simulation_result.get("home_cover_prob")
                    away_cover_probability = simulation_result.get("away_cover_prob")
            except Exception as exc:
                errors.append(f"Monte Carlo simulation failed for game {game_id}: {exc}")

        nrfi_probability: float | None = None
        nrfi_recommendation = "NO DATA"
        nrfi_source = "unavailable"
        nrfi_fallback_reason = ""

        if (
            nrfi_model_available
            and nrfi_model is not None
            and extract_nrf_features is not None
        ):
            try:
                nrfi_features = extract_nrf_features(
                    game=game,
                    pitcher_data=pitcher_data,
                    features=features,
                )
                if isinstance(nrfi_features, pd.DataFrame):
                    nrfi_input = nrfi_features
                elif isinstance(nrfi_features, dict):
                    feature_columns = getattr(
                        nrfi_model,
                        "feature_cols",
                        list(nrfi_features.keys()),
                    )
                    nrfi_input = pd.DataFrame(
                        [
                            [
                                nrfi_features.get(column, 0.0)
                                for column in feature_columns
                            ]
                        ],
                        columns=feature_columns,
                    )
                else:
                    raise TypeError("Unsupported NRFI feature format.")

                nrfi_probability = float(
                    nrfi_model.predict_proba(nrfi_input)[0, 1]
                )
                nrfi_recommendation = (
                    "NRFI" if nrfi_probability >= 0.55 else "NO BET"
                )
                nrfi_source = "ml"
            except Exception as exc:
                nrfi_fallback_reason = f"NRFI ML failed: {exc}"

        if nrfi_source != "ml":
            (
                nrfi_probability,
                nrfi_recommendation,
                nrfi_source,
                manual_nrfi_reason,
            ) = calculate_manual_nrfi(game, pitcher_data)
            if not nrfi_fallback_reason:
                nrfi_fallback_reason = manual_nrfi_reason

        min_moneyline_edge = max(
            0.0,
            as_float(
                getattr(config, "MIN_MONEYLINE_EDGE", 0.03),
                0.03,
            ),
        )
        max_kelly_fraction = max(
            0.0,
            as_float(
                getattr(config, "MAX_KELLY_FRACTION", 0.025),
                0.025,
            ),
        )

        moneyline_gate_status = "TRACKING_ONLY_ODDS_UNAVAILABLE"
        moneyline_selected_side = ""
        moneyline_selected_edge: float | None = None

        if odds_are_usable:
            home_kelly = 0.0
            away_kelly = 0.0

            home_edge = model_edge_home
            away_edge = (
                -model_edge_home
                if model_edge_home is not None
                else None
            )

            if (
                home_edge is not None
                and home_edge >= min_moneyline_edge
            ):
                moneyline_recommendation = f"{home_team} ML"
                moneyline_selected_side = "home"
                moneyline_selected_edge = home_edge
                moneyline_gate_status = "PAPER_BET_EDGE_PASSED"
                recommendation_status = "PAPER_BET"
                home_kelly = min(
                    kelly_criterion(
                        premarket_model_home_prob,
                        home_odds,
                    ),
                    max_kelly_fraction,
                )

            elif (
                away_edge is not None
                and away_edge >= min_moneyline_edge
            ):
                moneyline_recommendation = f"{away_team} ML"
                moneyline_selected_side = "away"
                moneyline_selected_edge = away_edge
                moneyline_gate_status = "PAPER_BET_EDGE_PASSED"
                recommendation_status = "PAPER_BET"
                away_kelly = min(
                    kelly_criterion(
                        1 - premarket_model_home_prob,
                        away_odds,
                    ),
                    max_kelly_fraction,
                )

            else:
                moneyline_recommendation = "NO BET"
                moneyline_gate_status = "TRACKING_ONLY_EDGE_BELOW_THRESHOLD"
                recommendation_status = "TRACKING_ONLY"

            if (
                home_cover_probability is not None
                and as_float(home_cover_probability, 0.0) >= 0.55
            ):
                spread_recommendation = f"{home_team} spread"
            elif (
                away_cover_probability is not None
                and as_float(away_cover_probability, 0.0) >= 0.55
            ):
                spread_recommendation = f"{away_team} spread"
            else:
                spread_recommendation = "NO BET"

            if over_probability is None:
                total_recommendation = "NO DATA"
            elif as_float(over_probability, 0.5) >= 0.55:
                total_recommendation = "OVER"
            elif as_float(over_probability, 0.5) <= 0.45:
                total_recommendation = "UNDER"
            else:
                total_recommendation = "NO BET"

        else:
            home_kelly = 0.0
            away_kelly = 0.0
            moneyline_recommendation = "NO BET"
            spread_recommendation = "NO BET"
            total_recommendation = "NO BET"
            recommendation_status = "TRACKING_ONLY"

        recommendation_block_reason, recommendation_block_details = (
            build_recommendation_block_reason(
                recommendation_status=recommendation_status,
                moneyline_gate_status=moneyline_gate_status,
                odds_are_usable=odds_are_usable,
                odds_quality_status=odds_quality_status,
                suspicious_odds_reason=suspicious_odds_reason,
                market_probability=market_probability,
                model_edge_home=model_edge_home,
                moneyline_selected_edge=moneyline_selected_edge,
                min_moneyline_edge=min_moneyline_edge,
                model_source=model_source,
                model_error=model_error,
                daily_context_summary=daily_context_summary,
            )
        )

        betting_readiness = evaluate_betting_readiness(
            daily_context_summary,
            odds_quality_status=odds_quality_status,
            moneyline_gate_status=moneyline_gate_status,
            model_source=model_source,
            model_edge_home=model_edge_home,
            moneyline_selected_edge=moneyline_selected_edge,
            min_moneyline_edge=min_moneyline_edge,
            model_training_sample_count=clean_model_sample_count,
            production_sample_threshold=300,
        )

        live_bet_candidate = (
            recommendation_status == "PAPER_BET"
            and betting_readiness.get("effective_context_ready_for_betting") is True
            and as_float(betting_readiness.get("stake_multiplier"), 0.0) > 0
            and odds_quality_status == "OK"
            and str(model_source or "").strip().lower() == "ml"
        )

        recommendation_block_details.append(
            "Betting readiness: "
            f"{betting_readiness.get('betting_readiness_status')}, "
            f"score={as_float(betting_readiness.get('betting_readiness_score'), 0.0):.2f}, "
            f"stake_multiplier={as_float(betting_readiness.get('stake_multiplier'), 0.0):.2f}."
        )

        risk_flags = betting_readiness.get("betting_risk_flags") or []
        if risk_flags:
            recommendation_block_details.append(
                "Risk flags: " + ", ".join(str(flag) for flag in risk_flags) + "."
            )
        else:
            recommendation_block_details.append("Risk flags: none.")

        feature_health_flags = []

        if not savant_top3_available:
            feature_health_flags.append("savant_top3_unavailable")

        if features.get("top3_woba_diff", 0.0) == 0.0:
            feature_health_flags.append("top3_woba_zero")

        if (
            features.get("statcast_woba_diff", 0.0) == 0.0
            and features.get("statcast_barrel_diff", 0.0) == 0.0
            and features.get("statcast_hard_hit_diff", 0.0) == 0.0
        ):
            feature_health_flags.append("statcast_core_zero")

        selected_model_probability_for_guard = (
            premarket_model_home_prob
            if moneyline_selected_side == "home"
            else 1.0 - premarket_model_home_prob
            if moneyline_selected_side == "away"
            else premarket_model_home_prob
        )

        selected_market_probability_for_guard = (
            market_probability
            if moneyline_selected_side == "home"
            else 1.0 - market_probability
            if moneyline_selected_side == "away" and market_probability is not None
            else market_probability
        )

        risk_guard_context = {
            "lineup_status": daily_context_summary.get("lineup_status"),
            "feature_health_flags": feature_health_flags,
        }

        risk_guard_approved = False
        risk_guard_reason = "NOT_EVALUATED"

        if (
            recommendation_status == "PAPER_BET"
            and selected_market_probability_for_guard is not None
            and moneyline_selected_side in {"home", "away"}
        ):
            risk_guard_approved, risk_guard_reason = RISK_GUARD.validate_bet_candidate(
                context=risk_guard_context,
                model_prob=float(selected_model_probability_for_guard),
                market_no_vig_prob=float(selected_market_probability_for_guard),
                features=features,
            )
        elif recommendation_status == "PAPER_BET":
            risk_guard_reason = "REJECT: market probability or selected side unavailable"

        if recommendation_status == "PAPER_BET" and not risk_guard_approved:
            if "risk_guard_rejected" not in risk_flags:
                risk_flags.append("risk_guard_rejected")
            recommendation_block_details.append(
                f"Risk guard: {risk_guard_reason}."
            )

        live_bet_candidate = live_bet_candidate and bool(risk_guard_approved)

        prediction_item = {
            "game_id": game_id,
            "game_date": date_str,
            "start_time": start_time,
            "game_status": game_status,
            "home_team": home_team,
            "away_team": away_team,
            "pipeline_version": getattr(
                config,
                "PIPELINE_VERSION",
                "baseline_v2_clean",
            ),
            "snapshot_policy": getattr(
                config,
                "SNAPSHOT_POLICY",
                "first_seen_pregame",
            ),
            "betting_mode": getattr(
                config,
                "BETTING_MODE",
                "paper_trading",
            ),
            "rating_source": rating_source,
            "premarket_model_home_prob": round(premarket_model_home_prob, 4),
            "displayed_home_win_pct": round(displayed_home_win_pct, 4),
            "predicted_home_win_pct": round(displayed_home_win_pct, 4),
            "manual_no_odds_pred": round(no_odds_prediction, 4),
            "market_no_vig_home_prob": (
                round(market_probability, 4)
                if market_probability is not None
                else None
            ),
            "model_edge_home": (
                round(model_edge_home, 4)
                if model_edge_home is not None
                else None
            ),
            "model_disagreement_with_market": (
                round(abs(model_edge_home), 4)
                if model_edge_home is not None
                else None
            ),
            "edge_bucket": (
                "8pct_plus"
                if moneyline_selected_edge is not None
                and abs(moneyline_selected_edge) >= 0.08
                else "5_to_8pct"
                if moneyline_selected_edge is not None
                and abs(moneyline_selected_edge) >= 0.05
                else "3_to_5pct"
                if moneyline_selected_edge is not None
                and abs(moneyline_selected_edge) >= 0.03
                else "below_threshold"
            ),
            "risk_profile": (
                "blocked"
                if live_bet_candidate is False
                else "reduced_stake"
                if as_float(betting_readiness.get("stake_multiplier"), 0.0) < 1.0
                else "full_paper_stake"
            ),
            "missing_signal_flags": betting_readiness.get("betting_risk_flags", []),
            "model_source": model_source,
            "model_feature_count": len(model_features or []),
            "model_training_sample_count": clean_model_sample_count,
            "model_load_error": model_error,
            "home_moneyline_odds": home_odds,
            "away_moneyline_odds": away_odds,
            "spread_line": spread_line,
            "total_line": total_line,
            "odds_quality_status": odds_quality_status,
            "suspicious_odds_reason": suspicious_odds_reason,
            "odds_source": odds_source,
            "bookmaker_quotes": bookmaker_quotes,
            "market_adjustment_applied": market_adjustment_applied,
            "accuracy_safe_model_weight": round(accuracy_safe_model_weight, 4),
            "accuracy_safe_market_weight": round(accuracy_safe_market_weight, 4),
            "accuracy_safe_blend_reasons": accuracy_safe_blend_reasons,
            "recommendation_status": recommendation_status,
            "moneyline_gate_status": moneyline_gate_status,
            "risk_guard_approved": bool(risk_guard_approved),
            "risk_guard_reason": risk_guard_reason,
            "recommendation_block_reason": recommendation_block_reason,
            "recommendation_block_details": recommendation_block_details,
            "daily_context_summary": daily_context_summary,
            "weather_context_summary": weather_context_summary,
            "betting_readiness": betting_readiness,
            "effective_context_ready_for_betting": betting_readiness[
                "effective_context_ready_for_betting"
            ],
            "betting_readiness_status": betting_readiness[
                "betting_readiness_status"
            ],
            "betting_readiness_score": betting_readiness[
                "betting_readiness_score"
            ],
            "betting_risk_flags": sorted(set(risk_flags)),
            "stake_multiplier": betting_readiness["stake_multiplier"],
            "live_bet_candidate": live_bet_candidate,
            "moneyline_edge_threshold": round(min_moneyline_edge, 4),
            "max_kelly_fraction": round(max_kelly_fraction, 4),
            "moneyline_selected_side": moneyline_selected_side,
            "moneyline_selected_edge": (
                round(moneyline_selected_edge, 4)
                if moneyline_selected_edge is not None
                else None
            ),
            "moneyline_recommendation": moneyline_recommendation,
            "spread_recommendation": spread_recommendation,
            "total_recommendation": total_recommendation,
            "home_kelly_fraction": round(home_kelly, 4),
            "away_kelly_fraction": round(away_kelly, 4),
            "over_prob": (
                round(as_float(over_probability), 4)
                if over_probability is not None
                else None
            ),
            "home_cover_prob": (
                round(as_float(home_cover_probability), 4)
                if home_cover_probability is not None
                else None
            ),
            "away_cover_prob": (
                round(as_float(away_cover_probability), 4)
                if away_cover_probability is not None
                else None
            ),
            "nrfi_prob": (
                round(nrfi_probability, 4)
                if nrfi_probability is not None
                else None
            ),
            "nrfi_recommendation": nrfi_recommendation,
            "nrfi_source": nrfi_source,
            "nrfi_fallback_reason": nrfi_fallback_reason,
            "feature_health_flags": feature_health_flags,
            "savant_top3_available": bool(savant_top3_available),
            "features": {
                feature: round(as_float(features.get(feature), 0.0), 6)
                for feature in EXPECTED_FEATURES
            },
        }

        predictions.append(prediction_item)

    if refresh_opening_closing_flags is not None:
        try:
            market_odds_history_summary["opening_closing_refresh"] = (
                refresh_opening_closing_flags()
            )
        except Exception as exc:
            market_odds_history_summary["status"] = "partial_failure"
            market_odds_history_summary["errors"].append(
                f"Opening closing refresh failed: {exc}"
            )

    snapshot_storage_summary: dict[str, Any] = {
        "pipeline_version": getattr(
            config,
            "PIPELINE_VERSION",
            "baseline_v2_clean",
        ),
        "status": "not_attempted",
    }

    if append_first_seen_pregame_snapshots is None:
        snapshot_storage_summary["status"] = "unavailable"
        errors.append(
            "Snapshot storage unavailable: scripts.snapshot_store import failed."
        )
    else:
        try:
            snapshot_storage_summary = append_first_seen_pregame_snapshots(
                predictions
            )
            snapshot_storage_summary["status"] = "completed"
        except Exception as exc:
            snapshot_storage_summary = {
                "pipeline_version": getattr(
                    config,
                    "PIPELINE_VERSION",
                    "baseline_v2_clean",
                ),
                "status": "failed",
                "error": str(exc),
            }
            errors.append(f"Snapshot storage failed: {exc}")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "pipeline_version": getattr(
            config,
            "PIPELINE_VERSION",
            "baseline_v2_clean",
        ),
        "schedule_fetch_ok": schedule_fetch_ok,
        "scheduled_game_count": scheduled_game_count,
        "snapshot_storage_summary": snapshot_storage_summary,
        "market_odds_history_summary": market_odds_history_summary,
        "today_predictions": predictions,
        "errors": errors,
    }

    return output


def write_report(output: dict[str, Any]) -> None:
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    try:
        output = generate_predictions()
        write_report(output)
        print(
            "Prediction report written. "
            f"games={len(output.get('today_predictions', []))}, "
            f"schedule_fetch_ok={output.get('schedule_fetch_ok')}, "
            f"scheduled_game_count={output.get('scheduled_game_count')}, "
            f"errors={len(output.get('errors', []))}"
        )
    except Exception as exc:
        error_message = f"Fatal prediction error: {exc}"
        failure_output = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace(
                "+00:00",
                "Z",
            ),
            "schedule_fetch_ok": False,
            "scheduled_game_count": None,
            "today_predictions": [],
            "errors": [error_message, traceback.format_exc()],
        }
        write_report(failure_output)
        print(error_message)
        raise


if __name__ == "__main__":
    main()
