"""
Statcast top-3 hitter context client for MLB prediction pipeline.

Fetches recent Baseball Savant / Statcast data for the top-3 hitters in each
game's latest daily context snapshot, then produces per-game aggregated context
columns for prediction features.
"""

from __future__ import annotations

import io
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATCAST_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_SLEEP_SECONDS = 0.15
DEFAULT_TIMEOUT = 12
DEFAULT_MAX_UNIQUE_PLAYERS = 90
TOP3_PLAYER_CONTEXT_FILE = Path("data/top3_player_context.csv")

PLAYER_SUMMARY_FIELDS = [
    "player_id",
    "savant_available",
    "savant_error",
    "sample_pitches",
    "sample_batted_balls",
    "sample_events",
    "xwoba",
    "woba_value",
    "avg_launch_speed",
    "avg_launch_angle",
    "hard_hit_rate",
    "barrel_rate",
    "sweet_spot_rate",
    "whiff_rate",
    "k_rate_proxy",
    "bb_rate_proxy",
]

TOP3_CONTEXT_COLUMNS = [
    "game_id",
    "game_date",
    "captured_at",
    "lookback_days",
    "start_date",
    "end_date",
    "home_top3_player_ids",
    "away_top3_player_ids",
    "home_top3_sample_batted_balls",
    "away_top3_sample_batted_balls",
    "home_top3_xwoba",
    "away_top3_xwoba",
    "top3_xwoba_diff",
    "home_top3_woba",
    "away_top3_woba",
    "top3_woba_diff",
    "home_top3_hard_hit_rate",
    "away_top3_hard_hit_rate",
    "top3_hard_hit_rate_diff",
    "home_top3_barrel_rate",
    "away_top3_barrel_rate",
    "top3_barrel_rate_diff",
    "home_top3_avg_launch_speed",
    "away_top3_avg_launch_speed",
    "top3_avg_launch_speed_diff",
    "home_top3_savant_available_count",
    "away_top3_savant_available_count",
    "savant_top3_errors",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    """Return float or None, avoiding NaN and infinity."""
    try:
        if value is None:
            return None

        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None

        return float(number)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Return int or None, avoiding NaN and infinity."""
    try:
        if value is None:
            return None

        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None

        return int(number)
    except (TypeError, ValueError):
        return None


def _is_missing(value: Any) -> bool:
    """Return True if value is None, NaN, or an empty/null-like string."""
    if value is None:
        return True

    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)

    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null"}

    return False


def _clean_output_value(value: Any) -> Any:
    """Clean values before CSV output so NaN/inf/null strings do not leak."""
    if value is None:
        return ""

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""

    if isinstance(value, str) and value.strip().lower() in {"nan", "none", "null"}:
        return ""

    return value


def _clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply output cleaning to a DataFrame with pandas-version fallback."""
    frame = frame.copy()
    frame = frame.where(pd.notnull(frame), "")

    try:
        return frame.map(_clean_output_value)
    except AttributeError:
        return frame.applymap(_clean_output_value)


def _empty_player_summary(player_id: Any, error_msg: str = "") -> Dict[str, Any]:
    """Return a complete unavailable player summary."""
    player_id_int = _safe_int(player_id)

    return {
        "player_id": player_id_int if player_id_int is not None else "",
        "savant_available": False,
        "savant_error": str(error_msg or ""),
        "sample_pitches": 0,
        "sample_batted_balls": 0,
        "sample_events": 0,
        "xwoba": None,
        "woba_value": None,
        "avg_launch_speed": None,
        "avg_launch_angle": None,
        "hard_hit_rate": None,
        "barrel_rate": None,
        "sweet_spot_rate": None,
        "whiff_rate": None,
        "k_rate_proxy": None,
        "bb_rate_proxy": None,
    }


def parse_player_ids(value: Any) -> List[int]:
    """Parse comma-separated IDs, list values, numeric values, or empty values."""
    if _is_missing(value):
        return []

    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = str(value).split(",")

    player_ids: List[int] = []

    for item in raw_items:
        player_id = _safe_int(str(item).strip())
        if player_id is not None:
            player_ids.append(player_id)

    return player_ids


def _weighted_mean(values: List[Tuple[Optional[float], Optional[float]]]) -> Optional[float]:
    """Compute weighted mean from (value, weight) tuples."""
    weighted_sum = 0.0
    total_weight = 0.0

    for value, weight in values:
        value_float = _safe_float(value)
        weight_float = _safe_float(weight)

        if value_float is None or weight_float is None or weight_float <= 0:
            continue

        weighted_sum += value_float * weight_float
        total_weight += weight_float

    if total_weight <= 0:
        return None

    return weighted_sum / total_weight


def _safe_diff(home_value: Optional[float], away_value: Optional[float]) -> float:
    """Return home - away if both sides exist, else 0.0."""
    home_float = _safe_float(home_value)
    away_float = _safe_float(away_value)

    if home_float is None or away_float is None:
        return 0.0

    return float(home_float - away_float)


def _build_statcast_params(player_id: int, start_date: str, end_date: str) -> Dict[str, str]:
    """Build conservative Baseball Savant Statcast Search CSV params."""
    return {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfGT": "R|",
        "hfPR": "",
        "hfZ": "",
        "stadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": "",
        "hfSit": "",
        "player_type": "batter",
        "hfOuts": "",
        "opponent": "",
        "pitcher_throws": "",
        "batter_stands": "",
        "hfSA": "",
        "game_date_gt": start_date,
        "game_date_lt": end_date,
        "team": "",
        "position": "",
        "hfRO": "",
        "home_road": "",
        "hfFlag": "",
        "hfBBT": "",
        "metric_1": "",
        "hfInn": "",
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
        "sort_col": "pitches",
        "player_event_sort": "h_launch_speed",
        "sort_order": "desc",
        "min_pas": "0",
        "type": "details",
        "player_id": str(player_id),
    }


def _fetch_statcast_csv(
    player_id: int,
    start_date: str,
    end_date: str,
    timeout: int,
) -> Tuple[Optional[str], str, bool]:
    """Fetch Baseball Savant CSV text.

    Returns:
        (csv_text, error_message, is_request_error)

    No-data responses are not considered request errors.
    """
    params = _build_statcast_params(player_id, start_date, end_date)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; MLBPredictionApp/1.0; "
            "+https://github.com/wujoshnjr/mlb-prediction-app)"
        ),
        "Accept": "text/csv,text/plain,*/*",
    }

    try:
        response = requests.get(
            STATCAST_CSV_URL,
            params=params,
            headers=headers,
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        return None, f"Request failed for player {player_id}: {exc}", True

    if response.status_code != 200:
        return None, f"HTTP {response.status_code} for player {player_id}", True

    text = response.text or ""
    stripped = text.strip()

    if not stripped:
        return None, f"No data from Savant for player {player_id}", False

    lower_sample = stripped[:500].lower()
    if stripped.startswith("<") or "<html" in lower_sample or "<!doctype html" in lower_sample:
        return None, f"Savant returned HTML/no CSV for player {player_id}", False

    if "no records found" in lower_sample:
        return None, f"No data from Savant for player {player_id}", False

    return text, "", False


def _find_xwoba_column(columns: List[str]) -> Optional[str]:
    """Find a likely xwOBA / estimated wOBA column."""
    preferred = [
        "estimated_woba_using_speedangle",
        "estimated_woba_using_speed_angle",
        "estimated_woba_using_speed_and_angle",
    ]

    lower_to_original = {column.lower(): column for column in columns}

    for column in preferred:
        if column in lower_to_original:
            return lower_to_original[column]

    for column in columns:
        lowered = column.lower()
        if "estimated_woba" in lowered or "xwoba" in lowered:
            return column

    return None


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return numeric Series for column or empty numeric Series."""
    if column not in frame.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(frame[column], errors="coerce")


def _mean_or_none(series: pd.Series) -> Optional[float]:
    """Return mean of non-null values, or None."""
    clean = pd.to_numeric(series, errors="coerce").dropna()

    if clean.empty:
        return None

    return float(clean.mean())


def _compute_single_player_summary(
    player_id: int,
    csv_text: str,
) -> Dict[str, Any]:
    """Parse Statcast CSV text and compute one batter's summary metrics."""
    try:
        frame = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        return _empty_player_summary(player_id, f"Failed to parse CSV: {exc}")

    if frame.empty:
        return _empty_player_summary(player_id, "No records")

    sample_pitches = int(len(frame))

    launch_speed = _numeric_series(frame, "launch_speed")
    launch_angle = _numeric_series(frame, "launch_angle")

    if not launch_speed.empty:
        batted_mask = launch_speed.notna()
    else:
        batted_mask = pd.Series(False, index=frame.index)

    sample_batted_balls = int(batted_mask.sum())

    if "events" in frame.columns:
        event_text = frame["events"].astype(str).str.lower()
        event_mask = frame["events"].notna() & event_text.ne("nan")
    else:
        event_text = pd.Series("", index=frame.index, dtype=str)
        event_mask = pd.Series(False, index=frame.index)

    sample_events = int(event_mask.sum())

    xwoba_column = _find_xwoba_column(list(frame.columns))
    xwoba = _mean_or_none(_numeric_series(frame, xwoba_column)) if xwoba_column else None

    woba_value = (
        _mean_or_none(_numeric_series(frame, "woba_value"))
        if "woba_value" in frame.columns
        else None
    )

    avg_launch_speed = (
        _mean_or_none(launch_speed[batted_mask])
        if sample_batted_balls > 0 and not launch_speed.empty
        else None
    )

    avg_launch_angle = (
        _mean_or_none(launch_angle[batted_mask])
        if sample_batted_balls > 0 and not launch_angle.empty
        else None
    )

    hard_hit_rate = None
    if sample_batted_balls > 0 and not launch_speed.empty:
        hard_hits = int((launch_speed[batted_mask] >= 95).sum())
        hard_hit_rate = float(hard_hits / sample_batted_balls)

    barrel_rate = None
    if "launch_speed_angle" in frame.columns and sample_batted_balls > 0:
        launch_speed_angle = pd.to_numeric(
            frame.loc[batted_mask, "launch_speed_angle"],
            errors="coerce",
        )
        barrels = int((launch_speed_angle == 6).sum())
        barrel_rate = float(barrels / sample_batted_balls)

    sweet_spot_rate = None
    if sample_batted_balls > 0 and not launch_angle.empty:
        batted_angles = launch_angle[batted_mask].dropna()
        if not batted_angles.empty:
            sweet_spots = int(batted_angles.between(8, 32).sum())
            sweet_spot_rate = float(sweet_spots / len(batted_angles))

    whiff_rate = None
    if "description" in frame.columns and sample_pitches > 0:
        descriptions = frame["description"].astype(str).str.lower()
        whiffs = int(descriptions.str.contains("swinging_strike", na=False).sum())
        whiff_rate = float(whiffs / sample_pitches)

    k_rate_proxy = None
    bb_rate_proxy = None
    if sample_events > 0:
        events_only = event_text[event_mask]
        strikeouts = int(events_only.str.contains("strikeout", na=False).sum())
        walks = int(events_only.str.contains("walk", na=False).sum())

        k_rate_proxy = float(strikeouts / sample_events)
        bb_rate_proxy = float(walks / sample_events)

    return {
        "player_id": player_id,
        "savant_available": True,
        "savant_error": "",
        "sample_pitches": sample_pitches,
        "sample_batted_balls": sample_batted_balls,
        "sample_events": sample_events,
        "xwoba": xwoba,
        "woba_value": woba_value,
        "avg_launch_speed": avg_launch_speed,
        "avg_launch_angle": avg_launch_angle,
        "hard_hit_rate": hard_hit_rate,
        "barrel_rate": barrel_rate,
        "sweet_spot_rate": sweet_spot_rate,
        "whiff_rate": whiff_rate,
        "k_rate_proxy": k_rate_proxy,
        "bb_rate_proxy": bb_rate_proxy,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_batter_statcast_summary(
    player_id: Union[int, str],
    start_date: str,
    end_date: str,
    errors: Optional[List[str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Fetch recent Statcast CSV for one batter and compute summary metrics."""
    player_id_int = _safe_int(player_id)

    if player_id_int is None:
        message = f"Invalid player_id: {player_id}"
        if errors is not None:
            errors.append(message)
        return _empty_player_summary(player_id, message)

    if _is_missing(start_date) or _is_missing(end_date):
        return _empty_player_summary(
            player_id_int,
            "Missing start_date or end_date",
        )

    csv_text, fetch_error, is_request_error = _fetch_statcast_csv(
        player_id=player_id_int,
        start_date=str(start_date),
        end_date=str(end_date),
        timeout=timeout,
    )

    if csv_text is None:
        if errors is not None and is_request_error:
            errors.append(fetch_error)

        return _empty_player_summary(player_id_int, fetch_error)

    summary = _compute_single_player_summary(player_id_int, csv_text)

    if not summary.get("savant_available") and errors is not None:
        parse_error = str(summary.get("savant_error") or "")
        if parse_error and "No records" not in parse_error:
            errors.append(parse_error)

    return summary


def _aggregate_metric(
    summaries: List[Dict[str, Any]],
    metric_key: str,
    weight_key: str = "sample_batted_balls",
) -> Optional[float]:
    """Aggregate one metric using sample_batted_balls as weight."""
    weighted_values: List[Tuple[Optional[float], Optional[float]]] = []

    for summary in summaries:
        if not summary.get("savant_available"):
            continue

        value = _safe_float(summary.get(metric_key))
        weight = _safe_float(summary.get(weight_key))

        if value is not None and weight is not None and weight > 0:
            weighted_values.append((value, weight))

    return _weighted_mean(weighted_values)


def _sum_int(summaries: List[Dict[str, Any]], key: str) -> int:
    """Sum integer values from player summaries."""
    total = 0

    for summary in summaries:
        value = _safe_int(summary.get(key))
        if value is not None:
            total += value

    return int(total)


def _available_count(summaries: List[Dict[str, Any]]) -> int:
    """Count available Savant summaries."""
    return int(sum(1 for summary in summaries if bool(summary.get("savant_available"))))


def _build_empty_game_row(
    *,
    game_id: str,
    game_date: str,
    captured_at: str,
    lookback_days: int,
    start_date: str,
    end_date: str,
    home_ids: List[int],
    away_ids: List[int],
    game_errors: List[str],
) -> Dict[str, Any]:
    """Build a fallback per-game output row."""
    return {
        "game_id": game_id,
        "game_date": game_date,
        "captured_at": captured_at,
        "lookback_days": lookback_days,
        "start_date": start_date,
        "end_date": end_date,
        "home_top3_player_ids": ",".join(str(player_id) for player_id in home_ids),
        "away_top3_player_ids": ",".join(str(player_id) for player_id in away_ids),
        "home_top3_sample_batted_balls": 0,
        "away_top3_sample_batted_balls": 0,
        "home_top3_xwoba": None,
        "away_top3_xwoba": None,
        "top3_xwoba_diff": 0.0,
        "home_top3_woba": None,
        "away_top3_woba": None,
        "top3_woba_diff": 0.0,
        "home_top3_hard_hit_rate": None,
        "away_top3_hard_hit_rate": None,
        "top3_hard_hit_rate_diff": 0.0,
        "home_top3_barrel_rate": None,
        "away_top3_barrel_rate": None,
        "top3_barrel_rate_diff": 0.0,
        "home_top3_avg_launch_speed": None,
        "away_top3_avg_launch_speed": None,
        "top3_avg_launch_speed_diff": 0.0,
        "home_top3_savant_available_count": 0,
        "away_top3_savant_available_count": 0,
        "savant_top3_errors": "; ".join(game_errors),
    }


def _build_aggregated_game_row(
    *,
    game_id: str,
    game_date: str,
    captured_at: str,
    lookback_days: int,
    start_date: str,
    end_date: str,
    home_ids: List[int],
    away_ids: List[int],
    home_summaries: List[Dict[str, Any]],
    away_summaries: List[Dict[str, Any]],
    game_errors: List[str],
) -> Dict[str, Any]:
    """Build an aggregated per-game top-3 hitter row."""
    home_xwoba = _aggregate_metric(home_summaries, "xwoba")
    away_xwoba = _aggregate_metric(away_summaries, "xwoba")

    home_woba = _aggregate_metric(home_summaries, "woba_value")
    away_woba = _aggregate_metric(away_summaries, "woba_value")

    home_hard_hit = _aggregate_metric(home_summaries, "hard_hit_rate")
    away_hard_hit = _aggregate_metric(away_summaries, "hard_hit_rate")

    home_barrel = _aggregate_metric(home_summaries, "barrel_rate")
    away_barrel = _aggregate_metric(away_summaries, "barrel_rate")

    home_launch_speed = _aggregate_metric(home_summaries, "avg_launch_speed")
    away_launch_speed = _aggregate_metric(away_summaries, "avg_launch_speed")

    return {
        "game_id": game_id,
        "game_date": game_date,
        "captured_at": captured_at,
        "lookback_days": lookback_days,
        "start_date": start_date,
        "end_date": end_date,
        "home_top3_player_ids": ",".join(str(player_id) for player_id in home_ids),
        "away_top3_player_ids": ",".join(str(player_id) for player_id in away_ids),
        "home_top3_sample_batted_balls": _sum_int(home_summaries, "sample_batted_balls"),
        "away_top3_sample_batted_balls": _sum_int(away_summaries, "sample_batted_balls"),
        "home_top3_xwoba": home_xwoba,
        "away_top3_xwoba": away_xwoba,
        "top3_xwoba_diff": _safe_diff(home_xwoba, away_xwoba),
        "home_top3_woba": home_woba,
        "away_top3_woba": away_woba,
        "top3_woba_diff": _safe_diff(home_woba, away_woba),
        "home_top3_hard_hit_rate": home_hard_hit,
        "away_top3_hard_hit_rate": away_hard_hit,
        "top3_hard_hit_rate_diff": _safe_diff(home_hard_hit, away_hard_hit),
        "home_top3_barrel_rate": home_barrel,
        "away_top3_barrel_rate": away_barrel,
        "top3_barrel_rate_diff": _safe_diff(home_barrel, away_barrel),
        "home_top3_avg_launch_speed": home_launch_speed,
        "away_top3_avg_launch_speed": away_launch_speed,
        "top3_avg_launch_speed_diff": _safe_diff(home_launch_speed, away_launch_speed),
        "home_top3_savant_available_count": _available_count(home_summaries),
        "away_top3_savant_available_count": _available_count(away_summaries),
        "savant_top3_errors": "; ".join(game_errors),
    }


def _latest_context_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return latest row per game_id from daily context frame."""
    working = frame.copy()

    working["game_id"] = working["game_id"].astype(str)

    if "captured_at" in working.columns:
        working["captured_at_dt"] = pd.to_datetime(
            working["captured_at"],
            errors="coerce",
            utc=True,
        )
        working = working.sort_values(["game_id", "captured_at_dt"])
        return working.groupby("game_id", as_index=False).tail(1)

    return working.drop_duplicates(subset=["game_id"], keep="last")


def _load_top3_player_context_by_game(
    path: Union[str, Path] = TOP3_PLAYER_CONTEXT_FILE,
) -> Dict[str, Dict[str, Any]]:
    """Load top3 player source rows by game_id."""
    context_path = Path(path)
    if not context_path.exists():
        return {}

    try:
        frame = pd.read_csv(context_path)
    except Exception:
        return {}

    if frame.empty or "game_id" not in frame.columns:
        return {}

    working = frame.copy()
    working["game_id"] = working["game_id"].astype(str)

    if "top3_player_captured_at" in working.columns:
        working["top3_player_captured_at_dt"] = pd.to_datetime(
            working["top3_player_captured_at"],
            errors="coerce",
            utc=True,
        )
        if working["top3_player_captured_at_dt"].notna().any():
            working = working.sort_values(["game_id", "top3_player_captured_at_dt"])
            working = working.groupby("game_id", as_index=False).tail(1)

    return {
        str(row["game_id"]): row.to_dict()
        for _, row in working.iterrows()
    }


def build_savant_top3_context(
    daily_context_path: Union[str, Path] = "data/daily_game_context.csv",
    output_path: Union[str, Path, None] = "data/savant_top3_context.csv",
    as_of_date: Optional[str] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    errors: Optional[List[str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    max_unique_players: int = DEFAULT_MAX_UNIQUE_PLAYERS,
) -> Dict[str, Any]:
    """Build per-game Baseball Savant top-3 hitter context from daily context."""
    captured_at = _utc_now_iso()

    if errors is None:
        errors = []

    summary: Dict[str, Any] = {
        "status": "empty",
        "captured_at": captured_at,
        "daily_context_path": str(daily_context_path),
        "output_path": str(output_path) if output_path is not None else None,
        "games_processed": 0,
        "unique_players_requested": 0,
        "max_unique_players": int(max_unique_players),
        "player_summaries_available": 0,
        "rows_written": 0,
        "home_rows_with_top3_ids": 0,
        "away_rows_with_top3_ids": 0,
        "rows_with_any_savant": 0,
        "rows_with_both_sides_savant": 0,
        "errors": errors,
    }

    daily_path = Path(daily_context_path)

    if not daily_path.exists():
        summary["status"] = "failed"
        errors.append(f"Daily context file not found: {daily_path}")
        return summary

    try:
        context_frame = pd.read_csv(daily_path)
    except Exception as exc:
        summary["status"] = "failed"
        errors.append(f"Failed to read daily context CSV: {exc}")
        return summary

    if context_frame.empty:
        summary["status"] = "empty"
        return summary

    if "game_id" not in context_frame.columns:
        summary["status"] = "failed"
        errors.append("Daily context CSV missing game_id column")
        return summary

    if "game_date" not in context_frame.columns:
        summary["status"] = "failed"
        errors.append("Daily context CSV missing game_date column")
        return summary

    for column in ("home_top3_player_ids", "away_top3_player_ids"):
        if column not in context_frame.columns:
            context_frame[column] = ""

    if as_of_date is not None:
        context_frame = context_frame[
            context_frame["game_date"].astype(str).str[:10] == str(as_of_date)
        ]

    if context_frame.empty:
        summary["status"] = "empty"
        return summary

    latest_frame = _latest_context_rows(context_frame)
    top3_player_context_by_game = _load_top3_player_context_by_game()

    if latest_frame.empty:
        summary["status"] = "empty"
        return summary

    player_cache: Dict[int, Dict[str, Any]] = {}
    output_rows: List[Dict[str, Any]] = []

    for _, row in latest_frame.iterrows():
        game_id = str(row.get("game_id", ""))
        game_date = str(row.get("game_date", ""))[:10]

        top3_source_row = top3_player_context_by_game.get(game_id, {})

        home_ids = parse_player_ids(
            top3_source_row.get("home_top3_player_ids")
            or row.get("home_top3_player_ids")
        )
        away_ids = parse_player_ids(
            top3_source_row.get("away_top3_player_ids")
            or row.get("away_top3_player_ids")
        )

        if home_ids:
            summary["home_rows_with_top3_ids"] += 1
        if away_ids:
            summary["away_rows_with_top3_ids"] += 1

        game_errors: List[str] = []

        start_date = ""
        end_date = ""

        try:
            game_date_dt = datetime.strptime(game_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            start_date = (
                game_date_dt - timedelta(days=max(1, int(lookback_days)))
            ).strftime("%Y-%m-%d")
            end_date = game_date_dt.strftime("%Y-%m-%d")
        except Exception:
            game_errors.append(f"Invalid game_date '{game_date}'")

        if not start_date or not end_date:
            output_rows.append(
                _build_empty_game_row(
                    game_id=game_id,
                    game_date=game_date,
                    captured_at=captured_at,
                    lookback_days=lookback_days,
                    start_date=start_date,
                    end_date=end_date,
                    home_ids=home_ids,
                    away_ids=away_ids,
                    game_errors=game_errors,
                )
            )
            continue

        def get_side_summaries(player_ids: List[int]) -> List[Dict[str, Any]]:
            side_summaries: List[Dict[str, Any]] = []

            for player_id in player_ids[:3]:
                if player_id in player_cache:
                    summary_row = player_cache[player_id]
                else:
                    player_errors: List[str] = []

                    if len(player_cache) >= max_unique_players:
                        summary_row = _empty_player_summary(
                            player_id,
                            "Skipped by max_unique_players limit",
                        )
                    else:
                        summary_row = fetch_batter_statcast_summary(
                            player_id=player_id,
                            start_date=start_date,
                            end_date=end_date,
                            errors=player_errors,
                            timeout=timeout,
                        )

                    player_cache[player_id] = summary_row

                    for message in player_errors:
                        if message:
                            game_errors.append(message)

                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)

                side_summaries.append(summary_row)
                
                player_error = str(summary_row.get("savant_error") or "")
                if player_error:
                    game_errors.append(
                        f"player {player_id}: {player_error}"
                    )

            return side_summaries

        home_summaries = get_side_summaries(home_ids)
        away_summaries = get_side_summaries(away_ids)

        home_available = _available_count(home_summaries)
        away_available = _available_count(away_summaries)

        if home_available > 0 or away_available > 0:
            summary["rows_with_any_savant"] += 1
        if home_available > 0 and away_available > 0:
            summary["rows_with_both_sides_savant"] += 1

        output_rows.append(
            _build_aggregated_game_row(
                game_id=game_id,
                game_date=game_date,
                captured_at=captured_at,
                lookback_days=lookback_days,
                start_date=start_date,
                end_date=end_date,
                home_ids=home_ids,
                away_ids=away_ids,
                home_summaries=home_summaries,
                away_summaries=away_summaries,
                game_errors=game_errors,
            )
        )

    result_frame = pd.DataFrame(output_rows, columns=TOP3_CONTEXT_COLUMNS)
    result_frame = _clean_dataframe(result_frame)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result_frame.to_csv(destination, index=False)
        summary["rows_written"] = int(len(result_frame))

    summary["games_processed"] = int(len(latest_frame))
    summary["unique_players_requested"] = int(len(player_cache))
    summary["player_summaries_available"] = int(
        sum(1 for item in player_cache.values() if bool(item.get("savant_available")))
    )
    summary["status"] = "completed"

    return summary


def load_savant_top3_context(
    path: Union[str, Path] = "data/savant_top3_context.csv",
) -> pd.DataFrame:
    """Safely load Savant top-3 context CSV."""
    context_path = Path(path)

    if not context_path.exists():
        return pd.DataFrame(columns=TOP3_CONTEXT_COLUMNS)

    try:
        frame = pd.read_csv(context_path)
    except Exception:
        return pd.DataFrame(columns=TOP3_CONTEXT_COLUMNS)

    for column in TOP3_CONTEXT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    return frame[TOP3_CONTEXT_COLUMNS]


if __name__ == "__main__":
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cli_errors: List[str] = []

    cli_summary = build_savant_top3_context(
        daily_context_path="data/daily_game_context.csv",
        output_path="data/savant_top3_context.csv",
        as_of_date=today_utc,
        lookback_days=DEFAULT_LOOKBACK_DAYS,
        errors=cli_errors,
        timeout=DEFAULT_TIMEOUT,
        sleep_seconds=DEFAULT_SLEEP_SECONDS,
    )

    print(json.dumps(cli_summary, ensure_ascii=True, indent=2, default=str))
