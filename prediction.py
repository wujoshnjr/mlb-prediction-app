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
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import UnifiedSportsModel
from scripts.feature_schema import EXPECTED_FEATURES

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

REPORT_FILE = Path("report/prediction.json")
HISTORY_FILE = Path("data/historical_predictions.csv")
HISTORICAL_DIR = Path("data/historical")
LAST_GAME_FILE = Path("data/team_last_game.json")
MODEL_FILE = Path("data/calibrator.pkl")

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


def implied_prob(odds: float | None) -> float | None:
    if odds is None or odds <= 1:
        return None
    return 1 / (odds * 1.05)


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


def load_ml_model() -> tuple[Any | None, list[str] | None, str]:
    if not MODEL_FILE.exists():
        return None, None, "Model artifact does not exist."

    try:
        import joblib

        artifact = joblib.load(MODEL_FILE)
        if (
            isinstance(artifact, dict)
            and "model" in artifact
            and "features" in artifact
        ):
            print("Loaded ML model artifact and feature list.")
            return artifact["model"], list(artifact["features"]), ""

        return None, None, "Legacy model artifact is unsupported."
    except Exception as exc:
        return None, None, str(exc)


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

    ml_model, model_features, model_load_error = load_ml_model()
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

    historical_df, historical_count = load_historical_frames(errors)
    last_game_data = load_last_game_data()

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
                venue = str(game.get("venue", ""))
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
            implied_prob(home_odds)
            if odds_are_usable
            else None
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

        home_top3_woba = as_float(game.get("home_top3_avg_woba"), 0.320)
        away_top3_woba = as_float(game.get("away_top3_avg_woba"), 0.320)
        features["top3_woba_diff"] = home_top3_woba - away_top3_woba

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
                ml_weight = min(0.50, historical_count / 1000.0)
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

        market_adjustment_applied = False
        if market_probability is not None:
            predicted_home_win = (
                predicted_home_win * 0.90
                + market_probability * 0.10
            )
            market_adjustment_applied = True

        predicted_home_win = float(np.clip(predicted_home_win, 0.05, 0.95))

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

        if odds_are_usable:
            home_kelly = kelly_criterion(predicted_home_win, home_odds)
            away_kelly = kelly_criterion(1 - predicted_home_win, away_odds)

            if predicted_home_win >= 0.55:
                moneyline_recommendation = f"{home_team} ML"
            elif predicted_home_win <= 0.45:
                moneyline_recommendation = f"{away_team} ML"
            else:
                moneyline_recommendation = "NO BET"

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

            recommendation_status = "PAPER_BET"
        else:
            home_kelly = 0.0
            away_kelly = 0.0
            moneyline_recommendation = "NO BET"
            spread_recommendation = "NO BET"
            total_recommendation = "NO BET"
            recommendation_status = "TRACKING_ONLY"

        prediction_item = {
            "game_id": game_id,
            "game_date": date_str,
            "home_team": home_team,
            "away_team": away_team,
            "rating_source": rating_source,
            "predicted_home_win_pct": round(predicted_home_win, 4),
            "manual_no_odds_pred": round(no_odds_prediction, 4),
            "model_source": model_source,
            "model_feature_count": len(model_features or []),
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
            "recommendation_status": recommendation_status,
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
            "features": {
                feature: round(as_float(features.get(feature), 0.0), 6)
                for feature in EXPECTED_FEATURES
            },
        }

        predictions.append(prediction_item)
            output = {
        "generated_at": datetime.now().isoformat(),
        "schedule_fetch_ok": schedule_fetch_ok,
        "scheduled_game_count": scheduled_game_count,
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
            "generated_at": datetime.now().isoformat(),
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
