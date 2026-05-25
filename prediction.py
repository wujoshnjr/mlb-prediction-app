# prediction.py
"""Generate daily MLB predictions and write report/prediction.json.

This file preserves the original model pipeline while tightening the contracts
used by training and validation:
- ML inference uses the same shared feature schema as training.
- Schedule retrieval reports whether an empty slate is real or caused by a
  fetch failure.
- Experimental pitch-usage diagnostics stay outside the production ML schema
  until explicitly enabled and validated.
- JSON output is written as UTF-8 without escaping Chinese log text.
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
    """Import optional pipeline helpers without aborting baseline prediction."""
    try:
        module = importlib.import_module(module_name)
        return tuple(getattr(module, name) for name in names)
    except Exception as exc:  # optional dependency guard
        print(f"{module_name} å¯å¥å¤±æ: {exc}")
        return tuple(None for _ in names)


(MLBElosystem,) = optional_import("scripts.elo", "MLBElosystem")
(MonteCarloSimulator,) = optional_import("scripts.monte_carlo", "MonteCarloSimulator")
(calculate_catcher_effect,) = optional_import("scripts.catcher_utils", "calculate_catcher_effect")
(calculate_lag_features,) = optional_import("scripts.lag_features", "calculate_lag_features")
(calculate_pitcher_ratings,) = optional_import("scripts.player_ratings", "calculate_pitcher_ratings")
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
    "Arizona Diamondbacks": "D-backs", "Diamondbacks": "D-backs", "Arizona": "D-backs",
    "Atlanta Braves": "Braves", "Atlanta": "Braves",
    "Baltimore Orioles": "Orioles", "Baltimore": "Orioles",
    "Boston Red Sox": "Red Sox", "Boston": "Red Sox",
    "Chicago Cubs": "Cubs", "Chicago (NL)": "Cubs",
    "Chicago White Sox": "White Sox", "Chicago (AL)": "White Sox",
    "Cincinnati Reds": "Reds", "Cincinnati": "Reds",
    "Cleveland Guardians": "Guardians", "Cleveland": "Guardians",
    "Colorado Rockies": "Rockies", "Colorado": "Rockies",
    "Detroit Tigers": "Tigers", "Detroit": "Tigers",
    "Houston Astros": "Astros", "Houston": "Astros",
    "Kansas City Royals": "Royals", "Kansas City": "Royals",
    "Los Angeles Angels": "Angels", "Los Angeles (AL)": "Angels",
    "Los Angeles Dodgers": "Dodgers", "Los Angeles (NL)": "Dodgers",
    "Miami Marlins": "Marlins", "Miami": "Marlins",
    "Milwaukee Brewers": "Brewers", "Milwaukee": "Brewers",
    "Minnesota Twins": "Twins", "Minnesota": "Twins",
    "New York Mets": "Mets", "New York (NL)": "Mets",
    "New York Yankees": "Yankees", "New York (AL)": "Yankees",
    "Oakland Athletics": "Athletics", "Oakland": "Athletics", "Athletics": "Athletics",
    "Philadelphia Phillies": "Phillies", "Philadelphia": "Phillies",
    "Pittsburgh Pirates": "Pirates", "Pittsburgh": "Pirates",
    "San Diego Padres": "Padres", "San Diego": "Padres",
    "San Francisco Giants": "Giants", "San Francisco": "Giants",
    "Seattle Mariners": "Mariners", "Seattle": "Mariners",
    "St. Louis Cardinals": "Cardinals", "St. Louis": "Cardinals",
    "Tampa Bay Rays": "Rays", "Tampa Bay": "Rays",
    "Texas Rangers": "Rangers", "Texas": "Rangers",
    "Toronto Blue Jays": "Blue Jays", "Toronto": "Blue Jays",
    "Washington Nationals": "Nationals", "Washington": "Nationals",
}

TEAM_ID_MAP = {
    "Braves": 144, "Orioles": 110, "Red Sox": 111, "Cubs": 112,
    "White Sox": 145, "Reds": 113, "Guardians": 114, "Rockies": 115,
    "Tigers": 116, "Astros": 117, "Royals": 118, "Angels": 108,
    "Dodgers": 119, "Marlins": 146, "Brewers": 158, "Twins": 142,
    "Mets": 121, "Yankees": 147, "Athletics": 133, "Phillies": 143,
    "Pirates": 134, "Padres": 135, "Giants": 137, "Mariners": 136,
    "Cardinals": 138, "Rays": 139, "Rangers": 140, "Blue Jays": 141,
    "Nationals": 120, "D-backs": 109,
}

TEAM_TIMEZONES = {
    "Braves": "Eastern", "Orioles": "Eastern", "Red Sox": "Eastern",
    "Cubs": "Central", "White Sox": "Central", "Reds": "Eastern",
    "Guardians": "Eastern", "Rockies": "Mountain", "Tigers": "Eastern",
    "Astros": "Central", "Royals": "Central", "Angels": "Pacific",
    "Dodgers": "Pacific", "Marlins": "Eastern", "Brewers": "Central",
    "Twins": "Central", "Mets": "Eastern", "Yankees": "Eastern",
    "Athletics": "Pacific", "Phillies": "Eastern", "Pirates": "Eastern",
    "Padres": "Pacific", "Giants": "Pacific", "Mariners": "Pacific",
    "Cardinals": "Central", "Rays": "Eastern", "Rangers": "Central",
    "Blue Jays": "Eastern", "Nationals": "Eastern", "D-backs": "Mountain",
}

TIMEZONE_OFFSETS = {"Eastern": 0, "Central": -1, "Mountain": -2, "Pacific": -3}


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


def kelly_criterion(win_prob: float | None, odds: float | None, fraction: float = 0.25) -> float:
    if win_prob is None or odds is None or odds <= 1:
        return 0.0
    b = odds - 1
    raw_fraction = win_prob - (1 - win_prob) / b
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
        print(f"è®å {path} å¤±æ: {exc}")
        return {}


def load_historical_frames(errors: list[str]) -> tuple[pd.DataFrame | None, int]:
    parquet_frames: list[pd.DataFrame] = []
    if HISTORICAL_DIR.exists():
        for file_path in sorted(HISTORICAL_DIR.glob("*.parquet")):
            try:
                frame = pd.read_parquet(file_path).dropna(how="all")
                if not frame.empty:
                    parquet_frames.append(frame)
            except Exception as exc:
                errors.append(f"æ­·å² parquet è®åå¤±æ {file_path}: {exc}")
    historical_df = pd.concat(parquet_frames, ignore_index=True) if parquet_frames else None

    historical_count = 0
    if HISTORY_FILE.exists():
        try:
            history = pd.read_csv(HISTORY_FILE)
            if "home_win" in history.columns:
                historical_count = int(history["home_win"].notna().sum())
        except Exception as exc:
            errors.append(f"æ­·å²é æ¸¬æä»¶è®åå¤±æ: {exc}")
    return historical_df, historical_count


def schedule_diagnostics(schedule_df: pd.DataFrame, errors: list[str]) -> tuple[bool, int | None]:
    critical_schedule_tokens = ("mlb_statsapi fetch error", "mlb_statsapi module not loaded")
    schedule_failed = any(
        any(token in message.lower() for token in critical_schedule_tokens)
        for message in errors
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
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)
    games = frame["wins"] + frame["losses"]
    frame["win_pct"] = np.where(games > 0, frame["wins"] / games, 0.5)
    return frame


def load_ml_model() -> tuple[Any | None, list[str] | None, str]:
    if not MODEL_FILE.exists():
        return None, None, "æ¨¡åæªæ¡ä¸å­å¨"
    try:
        import joblib
        artifact = joblib.load(MODEL_FILE)
        if isinstance(artifact, dict) and "model" in artifact and "features" in artifact:
            print("å·²è¼å¥ ML æ¨¡ååç¹å¾µåè¡¨")
            return artifact["model"], list(artifact["features"]), ""
        return None, None, "èæ ¼å¼æ¨¡åä¸æ¯æ´"
    except Exception as exc:
        return None, None, str(exc)


def load_nrfi_model() -> tuple[Any | None, bool]:
    if not getattr(config, "NRFI_USE_ML", False) or NRFIModel is None:
        return None, False
    model_path = Path("models/nrf_model.pkl")
    if not model_path.exists():
        print("NRFI æ¨¡åæªæ¡ä¸å­å¨ï¼å°ä½¿ç¨æå·¥å¬å¼")
        return None, False
    try:
        nrfi_model = NRFIModel(str(model_path))
        nrfi_model.load()
        print("NRFI æ¨¡åå·²è¼å¥")
        return nrfi_model, True
    except Exception as exc:
        print(f"NRFI æ¨¡åè¼å¥å¤±æ: {exc}")
        return None, False


def generate_predictions(elo_system: Any | None = None) -> dict[str, Any]:
    print("éå§æåè³æ...")
    source = UnifiedSportsModel()
    data = source.gather_all_data()
    date_str = datetime.now().strftime("%Y-%m-%d")
    errors = [str(message) for message in data.get("errors", [])]

    model, model_features, model_load_error = load_ml_model()
    nrfi_model, nrfi_model_loaded = load_nrfi_model()
    last_game_dict = load_json_dict(LAST_GAME_FILE)

    if elo_system is None and MLBElosystem is not None:
        try:
            elo_system = MLBElosystem()
            print("ELO ç³»çµ±å·²è¼å¥")
        except Exception as exc:
            errors.append(f"ELO ç³»çµ±è¼å¥å¤±æ: {exc}")
            elo_system = None

    glicko_league = None
    if getattr(config, "RATINGS_ENGINE", "elo") == "glicko2" and load_glicko2_league is not None:
        try:
            glicko_league = load_glicko2_league()
            print("Glicko2 è©ç´ç³»çµ±å·²è¼å¥")
        except Exception as exc:
            errors.append(f"Glicko2 è¼å¥å¤±æ: {exc}")

    try:
        save_snapshot, = optional_import("scripts.elo_momentum", "save_elo_snapshot")
        if save_snapshot is not None:
            save_snapshot()
    except Exception as exc:
        errors.append(f"ELO å¿«ç§å²å­å¤±æ: {exc}")

    teams_df = prepare_team_frame(data.get("sportsipy_teams", []))
    team_rows = {str(row["name"]): row for _, row in teams_df.iterrows()}
    league_avg_runs = 4.5
    if not teams_df.empty:
        games_total = float((teams_df["wins"] + teams_df["losses"]).sum() / 2)
        if games_total > 0:
            league_avg_runs = float(teams_df["runs_scored"].sum() / games_total)
    pythag_exponent = league_avg_runs ** 0.287
    print(f"åæ Pythag ææ¸: {pythag_exponent:.3f}")

    bt_strengths: dict[str, float] = {}
    if get_bradley_terry_strengths is not None:
        try:
            bt_strengths = get_bradley_terry_strengths() or {}
        except Exception as exc:
            errors.append(f"Bradley-Terry å¤±æ: {exc}")

    odds_dict: dict[tuple[str, str], list[float]] = {}
    odds_df = pd.DataFrame(data.get("odds_data", []))
    if not odds_df.empty:
        for _, row in odds_df.iterrows():
            key = (normalize_team(row.get("home_team")), normalize_team(row.get("away_team")))
            odds_value = as_float(row.get("odds"), default=0.0)
            if odds_value > 1:
                odds_dict.setdefault(key, []).append(odds_value)
    else:
        errors.append("è³ çè³æçºç©ºï¼æ¬æ¬¡åä¿ç baseline é æ¸¬")

    schedule_df = pd.DataFrame(data.get("mlb_statsapi", []))
    schedule_fetch_ok, scheduled_game_count = schedule_diagnostics(schedule_df, errors)
    if not schedule_fetch_ok:
        print("è³½ç¨è³æåå¾å¤±æ")
    elif scheduled_game_count == 0:
        print("ç¶æ¥ç¢ºèªç¡æ¯è³½")
    else:
        print(f"ç¶æ¥æ¯è³½æ¸é: {scheduled_game_count}")

    pitchers_df = pd.DataFrame(data.get("pitchers", []))
    pitcher_dict: dict[Any, Any] = {}
    if not pitchers_df.empty and "game_id" in pitchers_df.columns:
        pitcher_dict = {row["game_id"]: row for _, row in pitchers_df.iterrows()}

    bullpen_df = pd.DataFrame(data.get("bullpen", []))
    bullpen_dict: dict[Any, Any] = {}
    if not bullpen_df.empty and "team_id" in bullpen_df.columns:
        bullpen_dict = {row["team_id"]: row for _, row in bullpen_df.iterrows()}
    bullpen_availability_dict: dict[Any, float] = {}
    if calculate_bullpen_availability is not None and not bullpen_df.empty:
        try:
            bullpen_availability_dict = calculate_bullpen_availability(bullpen_df) or {}
        except Exception as exc:
            errors.append(f"çæ£å¯ç¨æ§è¨ç®å¤±æ: {exc}")

    platoon_df = pd.DataFrame(data.get("platoon", []))
    platoon_dict: dict[str, dict[str, dict[str, float]]] = {}
    if not platoon_df.empty:
        for _, row in platoon_df.iterrows():
            team = normalize_team(row.get("team_name"))
            split = str(row.get("split", ""))
            platoon_dict.setdefault(team, {})[split] = {"ops": as_float(row.get("ops"), 0.700)}

    savant_df = pd.DataFrame(data.get("savant_statcast", []))
    print(f"Statcast è³æåæ¸: {len(savant_df)}")
    statcast_team_stats: dict[str, dict[str, float]] = {}
    if not savant_df.empty and "batter_team" in savant_df.columns and "launch_speed" in savant_df.columns:
        savant_df = savant_df.copy()
        savant_df["batter_team"] = savant_df["batter_team"].map(normalize_team)
        for column in ("launch_speed", "barrel", "hard_hit", "expected_woba"):
            if column not in savant_df.columns:
                savant_df[column] = np.nan
        grouped = savant_df.groupby("batter_team").agg(
            avg_launch_speed=("launch_speed", "mean"),
            barrel_rate=("barrel", "mean"),
            hard_hit_rate=("hard_hit", "mean"),
            avg_expected_woba=("expected_woba", "mean"),
        )
        for team, row in grouped.iterrows():
            statcast_team_stats[str(team)] = {
                "avg_launch_speed": as_float(row["avg_launch_speed"]),
                "barrel_rate": as_float(row["barrel_rate"]),
                "hard_hit_rate": as_float(row["hard_hit_rate"]),
                "avg_expected_woba": as_float(row["avg_expected_woba"]),
            }

    pitch_movement_dict: dict[str, float] = {}
    if not savant_df.empty and {"home_team", "pfx_x", "pfx_z"}.issubset(savant_df.columns):
        temp = savant_df[["home_team", "pfx_x", "pfx_z"]].copy()
        temp["home_team"] = temp["home_team"].map(normalize_team)
        temp = temp.dropna()
        for team, group in temp.groupby("home_team"):
            pitch_movement_dict[str(team)] = float(
                np.sqrt(group["pfx_x"].mean() ** 2 + group["pfx_z"].mean() ** 2)
            )

    def team_average(column: str) -> dict[str, float]:
        if savant_df.empty or "home_team" not in savant_df.columns or column not in savant_df.columns:
            return {}
        temp = savant_df[["home_team", column]].copy().dropna()
        temp["home_team"] = temp["home_team"].map(normalize_team)
        return {str(team): float(value) for team, value in temp.groupby("home_team")[column].mean().items()}

    bat_speed_dict = team_average("bat_speed")
    sprint_speed_dict = team_average("sprint_speed")
    swing_miss_dict = team_average("whiff")
    csw_dict = team_average("csw")

    barrel_against_dict: dict[str, float] = {}
    if not savant_df.empty and {"home_team", "barrel"}.issubset(savant_df.columns):
        temp = savant_df[["home_team", "barrel"]].copy()
        temp["home_team"] = temp["home_team"].map(normalize_team)
        temp["barrel"] = pd.to_numeric(temp["barrel"], errors="coerce").fillna(0.0)
        barrel_against_dict = {str(team): float(value) for team, value in temp.groupby("home_team")["barrel"].mean().items()}

    pitcher_rating_dict: dict[str, float] = {}
    if calculate_pitcher_ratings is not None and not savant_df.empty:
        try:
            pitcher_rating_dict = calculate_pitcher_ratings(savant_df) or {}
        except Exception as exc:
            errors.append(f"ææè©åå¤±æ: {exc}")

    weather_df = pd.DataFrame(data.get("openmeteo_weather", []))
    avg_wind_speed = as_float(weather_df["wind_speed"].mean(), 0.0) if "wind_speed" in weather_df else 0.0
    avg_wind_dir = as_float(weather_df["wind_direction"].mean(), 0.0) if "wind_direction" in weather_df else 0.0
    avg_temp = as_float(weather_df["temperature_2m"].mean(), 20.0) if "temperature_2m" in weather_df else 20.0
    avg_precip = as_float(weather_df["precipitation"].mean(), 0.0) if "precipitation" in weather_df else 0.0

    injuries_df = pd.DataFrame(data.get("injuries", []))
    injury_index: dict[str, float] = {}
    if not injuries_df.empty and {"team_name", "status"}.issubset(injuries_df.columns):
        def injury_severity(status: Any) -> float:
            value = str(status).lower()
            if "60-day" in value or "60 day" in value:
                return 2.0
            if "10-day" in value or "15-day" in value:
                return 1.0
            return 0.5
        injuries_df = injuries_df.copy()
        injuries_df["team_name"] = injuries_df["team_name"].map(normalize_team)
        injuries_df["severity_score"] = injuries_df["status"].map(injury_severity)
        injury_index = {
            str(team): float(score)
            for team, score in injuries_df.groupby("team_name")["severity_score"].sum().items()
        }

    umpire_df = pd.DataFrame(data.get("umpires", []))
    umpire_dict: dict[Any, Any] = {}
    if not umpire_df.empty and "game_id" in umpire_df.columns:
        umpire_dict = {row["game_id"]: row for _, row in umpire_df.iterrows()}

    historical_df, historical_count = load_historical_frames(errors)
    predictions: list[dict[str, Any]] = []

    for _, game in schedule_df.iterrows():
        home = normalize_team(game.get("home_team"))
        away = normalize_team(game.get("away_team"))
        if not home or not away or home == "Unknown" or away == "Unknown":
            errors.append(f"è·³é game_id={game.get('game_id')}: çéåç¨±ä¸å®æ´")
            continue
        if home not in team_rows or away not in team_rows:
            errors.append(f"ç¡æ³å»ºç«é æ¸¬ {away} @ {home}: ç¼ºå°çéæ°åè³æ")
            continue

        home_row = team_rows[home]
        away_row = team_rows[away]
        home_pct = as_float(home_row["win_pct"], 0.5)
        away_pct = as_float(away_row["win_pct"], 0.5)
        home_games = max(as_int(home_row["wins"]) + as_int(home_row["losses"]), 1)
        away_games = max(as_int(away_row["wins"]) + as_int(away_row["losses"]), 1)
        home_runs_per_game = as_float(home_row["runs_scored"], 400.0) / home_games
        away_runs_per_game = as_float(away_row["runs_scored"], 400.0) / away_games

        elo_diff = 0.0
        if elo_system is not None:
            elo_diff = as_float(elo_system.elos.get(home, 1500), 1500.0) - as_float(elo_system.elos.get(away, 1500), 1500.0) + as_float(getattr(elo_system, "home_adv", 24.0), 24.0)
        bt_strength_diff = as_float(bt_strengths.get(home), 0.0) - as_float(bt_strengths.get(away), 0.0)

        elo_momentum_7d = 0.0
        elo_momentum_30d = 0.0
        if get_elo_momentum is not None:
            try:
                elo_momentum_7d = as_float(get_elo_momentum(home, 7)) - as_float(get_elo_momentum(away, 7))
                elo_momentum_30d = as_float(get_elo_momentum(home, 30)) - as_float(get_elo_momentum(away, 30))
            except Exception as exc:
                errors.append(f"ELO åéå¤±æ: {exc}")

        home_odds_values = odds_dict.get((home, away), [])
        home_odds = float(np.mean(home_odds_values)) if home_odds_values else None
        market_prob = implied_prob(home_odds) or 0.5

        pitcher_data = pitcher_dict.get(game.get("game_id"))
        sp_era_diff = sp_fip_diff = sp_stuff_plus_diff = sp_csw_diff = 0.0
        home_sp_era = away_sp_era = 4.5
        home_pitch_hand = away_pitch_hand = "R"
        if pitcher_data is not None:
            try:
                home_sp_era = as_float(pitcher_data.get("home_era"), 4.5)
                away_sp_era = as_float(pitcher_data.get("away_era"), 4.5)
                sp_era_diff = home_sp_era - away_sp_era
                sp_fip_diff = as_float(pitcher_data.get("home_fip"), 4.0) - as_float(pitcher_data.get("away_fip"), 4.0)
                sp_stuff_plus_diff = as_float(pitcher_data.get("home_stuff_plus"), 100.0) - as_float(pitcher_data.get("away_stuff_plus"), 100.0)
                sp_csw_diff = as_float(pitcher_data.get("home_csw_pct"), 0.28) - as_float(pitcher_data.get("away_csw_pct"), 0.28)
                home_pitch_hand = str(pitcher_data.get("home_pitch_hand", "R"))
                away_pitch_hand = str(pitcher_data.get("away_pitch_hand", "R"))
            except Exception as exc:
                errors.append(f"è§£æææè³æå¤±æ: {exc}")

        home_rest = away_rest = 2
        today = datetime.strptime(date_str, "%Y-%m-%d")
        try:
            if home in last_game_dict:
                home_rest = max(0, (today - datetime.strptime(str(last_game_dict[home]), "%Y-%m-%d")).days - 1)
            if away in last_game_dict:
                away_rest = max(0, (today - datetime.strptime(str(last_game_dict[away]), "%Y-%m-%d")).days - 1)
        except Exception as exc:
            errors.append(f"ä¼æ¯æ¥è¨ç®å¤±æ: {exc}")
        rest_diff = home_rest - away_rest
        timezone_diff = TIMEZONE_OFFSETS.get(TEAM_TIMEZONES.get(away, "Eastern"), 0) - TIMEZONE_OFFSETS.get(TEAM_TIMEZONES.get(home, "Eastern"), 0)
        is_day_game = as_int(game.get("is_day_game"), 0)

        home_id = TEAM_ID_MAP.get(home)
        away_id = TEAM_ID_MAP.get(away)
        home_bullpen = bullpen_dict.get(home_id, {})
        away_bullpen = bullpen_dict.get(away_id, {})
        bullpen_ip_diff = as_float(home_bullpen.get("bullpen_innings")) - as_float(away_bullpen.get("bullpen_innings"))
        back2back_diff = as_int(home_bullpen.get("back_to_back")) - as_int(away_bullpen.get("back_to_back"))
        bullpen_availability_diff = as_float(bullpen_availability_dict.get(home_id), 50.0) - as_float(bullpen_availability_dict.get(away_id), 50.0)

        static_park = 1.0
        if get_park_factor is not None:
            try:
                static_park = as_float(get_park_factor(game.get("venue", "")), 1.0)
            except Exception as exc:
                errors.append(f"çå ´ä¿æ¸å¤±æ: {exc}")
        temp_fahrenheit = avg_temp * 9 / 5 + 32
        park_temp_effect = (temp_fahrenheit - 70) / 10 * 0.01
        dynamic_park_factor = static_park * (1 + park_temp_effect)

        home_split = "vsLhp" if home_pitch_hand == "L" else "vsRhp"
        away_split = "vsLhp" if away_pitch_hand == "L" else "vsRhp"
        platoon_ops_diff = as_float(platoon_dict.get(home, {}).get(home_split, {}).get("ops"), 0.700) - as_float(platoon_dict.get(away, {}).get(away_split, {}).get("ops"), 0.700)

        catcher_era_diff = cs_diff = 0.0
        if calculate_catcher_effect is not None and game.get("home_catcher_id") and game.get("away_catcher_id"):
            try:
                catcher_era_diff, cs_diff = calculate_catcher_effect(game.get("home_catcher_id"), game.get("away_catcher_id"), int(date_str[:4]))
            except Exception as exc:
                errors.append(f"ææææå¤±æ: {exc}")

        home_stat = statcast_team_stats.get(home, {})
        away_stat = statcast_team_stats.get(away, {})
        statcast_launch_speed_diff = as_float(home_stat.get("avg_launch_speed")) - as_float(away_stat.get("avg_launch_speed"))
        statcast_barrel_diff = as_float(home_stat.get("barrel_rate")) - as_float(away_stat.get("barrel_rate"))
        statcast_hard_hit_diff = as_float(home_stat.get("hard_hit_rate")) - as_float(away_stat.get("hard_hit_rate"))
        statcast_woba_diff = as_float(home_stat.get("avg_expected_woba")) - as_float(away_stat.get("avg_expected_woba"))
        pitch_movement_diff = as_float(pitch_movement_dict.get(home)) - as_float(pitch_movement_dict.get(away))
        k_pct_diff = as_float(home_row["home_k_pct"]) - as_float(away_row["away_k_pct"])
        bb_pct_diff = as_float(home_row["home_bb_pct"]) - as_float(away_row["away_bb_pct"])
        avg_bat_speed_diff = as_float(bat_speed_dict.get(home)) - as_float(bat_speed_dict.get(away))
        sprint_speed_diff = as_float(sprint_speed_dict.get(home)) - as_float(sprint_speed_dict.get(away))
        pitcher_rating_diff = as_float(pitcher_rating_dict.get(home)) - as_float(pitcher_rating_dict.get(away))
        swing_miss_diff = as_float(swing_miss_dict.get(home)) - as_float(swing_miss_dict.get(away))
        csw_diff = as_float(csw_dict.get(home)) - as_float(csw_dict.get(away))
        barrel_bb_pct_diff = as_float(barrel_against_dict.get(home)) - as_float(barrel_against_dict.get(away))

        pitch_type_matchup_score = 0.0
        if getattr(config, "FEATURE_USE_PITCH_MATCHUP", False) and get_pitch_type_matchup_score is not None and pitcher_data is not None:
            try:
                pitch_type_matchup_score = as_float(get_pitch_type_matchup_score(pitcher_data.get("home_pitcher_id"), pitcher_data.get("away_pitcher_id")))
            except Exception as exc:
                errors.append(f"çç¨®å°ä½å¤±æ: {exc}")

        home_top3_woba = as_float(game.get("home_top3_avg_woba"), 0.320)
        away_top3_woba = as_float(game.get("away_top3_avg_woba"), 0.320)
        temp_effect = 0.02 * (avg_temp - 25) if avg_temp > 25 else 0.0
        precip_effect = -0.01 * avg_precip if avg_precip > 0 else 0.0
        wind_effect = 0.02 * avg_wind_speed * np.sin(np.radians(avg_wind_dir)) if avg_wind_speed > 10 else 0.0
        injury_diff = as_float(injury_index.get(home)) - as_float(injury_index.get(away))

        home_rs = as_float(home_row["runs_scored"], 400.0)
        home_ra = as_float(home_row["runs_allowed"], 400.0)
        away_rs = as_float(away_row["runs_scored"], 400.0)
        away_ra = as_float(away_row["runs_allowed"], 400.0)
        home_denominator = home_rs ** pythag_exponent + home_ra ** pythag_exponent
        away_denominator = away_rs ** pythag_exponent + away_ra ** pythag_exponent
        home_pythag = home_rs ** pythag_exponent / home_denominator if home_denominator else 0.5
        away_pythag = away_rs ** pythag_exponent / away_denominator if away_denominator else 0.5
        dynamic_pythag_diff = home_pythag - away_pythag
        log5_denominator = home_pct + away_pct - 2 * home_pct * away_pct
        log5_home = (home_pct - home_pct * away_pct) / log5_denominator if log5_denominator else 0.5

        lag30_winrate_diff = lag30_runs_diff = 0.0
        if calculate_lag_features is not None and historical_df is not None:
            try:
                lag30_winrate_diff, lag30_runs_diff = calculate_lag_features(home, away, historical_df, date_str, days=30)
            except Exception as exc:
                errors.append(f"æ»¯å¾ç¹å¾µå¤±æ: {exc}")

        odds_change = 0.0
        if HISTORY_FILE.exists() and home_odds is not None:
            try:
                history = pd.read_csv(HISTORY_FILE)
                if {"home_team", "away_team", "home_odds"}.issubset(history.columns):
                    last_odds = history[(history["home_team"] == home) & (history["away_team"] == away)]["home_odds"].dropna().tail(1).values
                    if len(last_odds):
                        odds_change = home_odds - as_float(last_odds[0])
            except Exception as exc:
                errors.append(f"è³ çè®åè¨ç®å¤±æ: {exc}")

        umpire_data = umpire_dict.get(game.get("game_id"), {})
        features = {key: 0.0 for key in EXPECTED_FEATURES}
        features.update({
            "elo_diff": round(elo_diff, 3),
            "sp_era_diff": round(sp_era_diff, 3),
            "sp_fip_diff": round(sp_fip_diff, 3),
            "sp_stuff_plus_diff": round(sp_stuff_plus_diff, 3),
            "sp_csw_diff": round(sp_csw_diff, 3),
            "bullpen_ip_diff": round(bullpen_ip_diff, 3),
            "rest_diff": float(rest_diff),
            "dynamic_park_factor": round(dynamic_park_factor, 3),
            "platoon_ops_diff": round(platoon_ops_diff, 3),
            "statcast_launch_speed_diff": round(statcast_launch_speed_diff, 3),
            "statcast_barrel_diff": round(statcast_barrel_diff, 3),
            "statcast_hard_hit_diff": round(statcast_hard_hit_diff, 3),
            "statcast_woba_diff": round(statcast_woba_diff, 3),
            "timezone_diff": float(timezone_diff),
            "is_day_game": float(is_day_game),
            "back2back_diff": float(back2back_diff),
            "catcher_era_diff": round(as_float(catcher_era_diff), 3),
            "cs_diff": round(as_float(cs_diff), 3),
            "wind_effect": round(float(wind_effect), 4),
            "temp_effect": round(float(temp_effect), 4),
            "precip_effect": round(float(precip_effect), 4),
            "injury_diff": round(injury_diff, 3),
            "dynamic_pythag_diff": round(dynamic_pythag_diff, 3),
            "log5_prob": round(log5_home, 3),
            "lag30_winrate_diff": round(as_float(lag30_winrate_diff), 3),
            "lag30_runs_diff": round(as_float(lag30_runs_diff), 3),
            "pitch_movement_diff": round(pitch_movement_diff, 3),
            "k_pct_diff": round(k_pct_diff, 3),
            "bb_pct_diff": round(bb_pct_diff, 3),
            "avg_bat_speed_diff": round(avg_bat_speed_diff, 3),
            "pitcher_rating_diff": round(pitcher_rating_diff, 3),
            "odds_change": round(odds_change, 4),
            "zone_size": round(as_float(umpire_data.get("zone_size"), 1.0), 3),
            "k_rate": round(as_float(umpire_data.get("k_rate"), 0.22), 3),
            "bullpen_availability_diff": round(bullpen_availability_diff, 3),
            "elo_momentum_7d": round(elo_momentum_7d, 3),
            "elo_momentum_30d": round(elo_momentum_30d, 3),
            "barrel_pa_diff": round(statcast_barrel_diff, 3),
            "hardhit_pa_diff": round(statcast_hard_hit_diff, 3),
            "swing_miss_diff": round(swing_miss_diff, 3),
            "csw_diff": round(csw_diff, 3),
            "barrel_bb_pct_diff": round(barrel_bb_pct_diff, 3),
            "sprint_speed_diff": round(sprint_speed_diff, 3),
            "pitch_type_matchup_score": round(pitch_type_matchup_score, 3),
            "top3_woba_diff": round(home_top3_woba - away_top3_woba, 3),
            "winrate_diff": round(home_pct - away_pct, 3),
            "bt_strength_diff": round(bt_strength_diff, 3),
        })

        diagnostics: dict[str, Any] = {}
        if getattr(config, "FEATURE_USE_PITCH_USAGE", False) and compute_pitch_usage_features is not None and not savant_df.empty:
            try:
                diagnostics["pitch_usage"] = compute_pitch_usage_features(savant_df, home, away, game.get("home_pitcher_id"), game.get("away_pitcher_id"))
            except Exception as exc:
                errors.append(f"Pitch Usage è¨ºæ·å¤±æ: {exc}")

        if getattr(config, "RATINGS_ENGINE", "elo") == "glicko2" and glicko_league is not None:
            try:
                rating_diff, rd_sum = glicko_league.get_rating_diff(home, away)
                features["elo_diff"] = round(as_float(rating_diff), 3)
                diagnostics["glicko_rd_sum"] = round(as_float(rd_sum), 3)
            except Exception as exc:
                errors.append(f"Glicko2 è©åå¤±æ: {exc}")

        neutral_elo_diff = features["elo_diff"]
        if getattr(config, "RATINGS_ENGINE", "elo") == "elo" and elo_system is not None:
            neutral_elo_diff -= as_float(getattr(elo_system, "home_adv", 24.0), 24.0)
        elo_available = elo_system is not None or getattr(config, "RATINGS_ENGINE", "elo") == "glicko2"
        elo_prob = 1 / (1 + 10 ** (-neutral_elo_diff / 400)) if elo_available else 0.5
        manual_pred = float(np.clip(elo_prob - 0.03 * np.clip(sp_era_diff, -2.0, 2.0), 0.05, 0.95))

        ml_pred: float | None = None
        ml_feature_list = model_features or EXPECTED_FEATURES
        if model is not None:
            try:
                feature_array = np.array([[features.get(feature, 0.0) for feature in ml_feature_list]], dtype=float)
                ml_pred = float(np.clip(model.predict_proba(feature_array)[0, 1], 0.05, 0.95))
            except Exception as exc:
                errors.append(f"ML é æ¸¬å¤±æ: {exc}")
        model_source = "ml" if ml_pred is not None else "manual"
        pred_home = ((1 - min(0.5, historical_count / 1000)) * manual_pred + min(0.5, historical_count / 1000) * ml_pred) if ml_pred is not None and historical_count > 100 else manual_pred
        pred_home += get_season_phase_adjustment(date_str, pred_home)
        pred_home = float(np.clip(pred_home * 0.90 + market_prob * 0.10, 0.05, 0.95))
        pred_away = 1 - pred_home

        over_prob = under_prob = home_cover = away_cover = None
        if MonteCarloSimulator is not None:
            try:
                home_off = home_runs_per_game / 4.5
                away_off = away_runs_per_game / 4.5
                home_pitch_factor = 4.5 / max(away_sp_era, 1.0)
                away_pitch_factor = 4.5 / max(home_sp_era, 1.0)
                expected_home_runs = float(np.clip(4.5 * home_off * home_pitch_factor * dynamic_park_factor, 2.5, 7.0))
                expected_away_runs = float(np.clip(4.5 * away_off * away_pitch_factor * dynamic_park_factor, 2.5, 7.0))
                simulator = MonteCarloSimulator(expected_home_runs, expected_away_runs, n_simulations=5000)
                simulator.simulate()
                home_cover, away_cover = simulator.spread_prob(-1.5)
                over_prob, under_prob, _ = simulator.total_prob(8.5)
            except Exception as exc:
                errors.append(f"Monte Carlo å¤±æ: {exc}")

        moneyline_recommendation = "PASS"
        if home_odds is not None and kelly_criterion(pred_home, home_odds) > 0.05:
            moneyline_recommendation = f"Bet {home} ({pred_home:.1%})"
        spread_recommendation = "PASS"
        if home_cover is not None and home_cover > 0.55:
            spread_recommendation = f"Bet {home} -1.5 ({home_cover:.1%})"
        elif away_cover is not None and away_cover > 0.55:
            spread_recommendation = f"Bet {away} +1.5 ({away_cover:.1%})"
        total_recommendation = "PASS"
        if over_prob is not None and over_prob > 0.55:
            total_recommendation = f"Bet OVER 8.5 ({over_prob:.1%})"
        elif under_prob is not None and under_prob > 0.55:
            total_recommendation = f"Bet UNDER 8.5 ({under_prob:.1%})"

        nrfi_prob: float | None = None
        nrfi_source = "unavailable"
        nrfi_fallback_reason: str | None = None
        if nrfi_model_loaded and nrfi_model is not None and extract_nrf_features is not None:
            try:
                nrfi_inputs = {
                    "home_first_era": as_float(pitcher_data.get("home_first_era") if pitcher_data is not None else None, 4.5),
                    "away_first_era": as_float(pitcher_data.get("away_first_era") if pitcher_data is not None else None, 4.5),
                    "home_top3_woba": home_top3_woba,
                    "away_top3_woba": away_top3_woba,
                }
                nrfi_features = extract_nrf_features(nrfi_inputs)
                feature_frame = pd.DataFrame([nrfi_features])[nrfi_model.feature_cols]
                nrfi_prob = float(nrfi_model.predict_proba(feature_frame)[0])
                nrfi_source = "ml"
            except Exception as exc:
                nrfi_fallback_reason = f"NRFI ML é æ¸¬å¤±æ: {exc}"
        if nrfi_prob is None and pitcher_data is not None:
            first_home = pitcher_data.get("home_first_era")
            first_away = pitcher_data.get("away_first_era")
            if pd.notna(first_home) and pd.notna(first_away) and home_top3_woba != 0.320 and away_top3_woba != 0.320:
                average_first_era = (as_float(first_home, 4.5) + as_float(first_away, 4.5)) / 2
                top3_factor = 1.0 - (home_top3_woba - 0.320) * 0.5 + (away_top3_woba - 0.320) * 0.5
                base_nrfi = max(0.3, min(0.7, 0.5 + (4.5 - average_first_era) * 0.08))
                nrfi_prob = float(np.clip(base_nrfi * top3_factor, 0.25, 0.75))
                nrfi_source = "manual"
        if nrfi_prob is None and nrfi_fallback_reason is None:
            nrfi_fallback_reason = "ç¼ºå°ä¸å± ERA æåä¸æ£ wOBA è³æ"
        nrfi_recommendation = "NO DATA" if nrfi_prob is None else (f"NRFI ({nrfi_prob:.1%})" if nrfi_prob > 0.55 else f"YRFI ({1 - nrfi_prob:.1%})")

        prediction: dict[str, Any] = {
            "game_id": game.get("game_id"),
            "game_date": game.get("game_date", date_str),
            "home_team": home,
            "away_team": away,
            "predicted_home_win_pct": round(pred_home, 3),
            "predicted_away_win_pct": round(pred_away, 3),
            "home_odds": home_odds,
            "moneyline_recommendation": moneyline_recommendation,
            "spread_recommendation": spread_recommendation,
            "total_recommendation": total_recommendation,
            "nrfi_recommendation": nrfi_recommendation,
            "nrfi_prob": round(nrfi_prob, 3) if nrfi_prob is not None else None,
            "nrfi_source": nrfi_source,
            "nrfi_fallback_reason": nrfi_fallback_reason,
            "over_prob": round(as_float(over_prob), 3) if over_prob is not None else None,
            "under_prob": round(as_float(under_prob), 3) if under_prob is not None else None,
            "home_cover_prob": round(as_float(home_cover), 3) if home_cover is not None else None,
            "away_cover_prob": round(as_float(away_cover), 3) if away_cover is not None else None,
            "model_source": model_source,
            "model_feature_count": len(ml_feature_list) if model is not None else 0,
            "model_load_error": model_load_error if model is None else "",
        }
        if diagnostics:
            prediction["diagnostics"] = diagnostics
        predictions.append(prediction)

    if predictions:
        errors.append(f"å¹³åä¸»éæ¦ç: {np.mean([item['predicted_home_win_pct'] for item in predictions]):.3f}")

    output = {
        "generated_at": datetime.now().isoformat(),
        "schedule_fetch_ok": schedule_fetch_ok,
        "scheduled_game_count": scheduled_game_count,
        "today_predictions": predictions,
        "errors": errors,
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False, default=str)
    print("prediction.json å·²çæ")
    return output


if __name__ == "__main__":
    try:
        generate_predictions()
    except Exception as exc:
        print(f"å´éé¯èª¤: {exc}")
        traceback.print_exc()
        raise SystemExit(1)
