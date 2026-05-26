"""Unified upstream data collection with a stable prediction.py contract."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd


fetch_mlb_statsapi = None
fetch_savant_statcast = None
fetch_retrosheet = None
fetch_pybaseball = None
fetch_sportsipy = None
fetch_openmeteo = None
fetch_balldontlie = None
fetch_odds = None
fetch_probable_pitchers = None
fetch_injuries = None
fetch_bullpen_stats = None
fetch_platoon_splits = None
fetch_umpire_data = None

try:
    from scripts.mlb_stats_client import fetch_mlb_statsapi
except Exception as exc:
    print(f"Warning: failed to import mlb_stats_client: {exc}")

try:
    from scripts.savant_client import fetch_savant_statcast
except Exception as exc:
    print(f"Warning: failed to import savant_client: {exc}")

try:
    from scripts.retro_client import fetch_retrosheet
except Exception as exc:
    print(f"Warning: failed to import retro_client: {exc}")

try:
    from scripts.pybaseball_client import fetch_pybaseball
except Exception as exc:
    print(f"Warning: failed to import pybaseball_client: {exc}")

try:
    from scripts.sportsipy_client import fetch_sportsipy
except Exception as exc:
    print(f"Warning: failed to import sportsipy_client: {exc}")

try:
    from scripts.openmeteo_client import fetch_openmeteo
except Exception as exc:
    print(f"Warning: failed to import openmeteo_client: {exc}")

try:
    from scripts.balldontlie_client import fetch_balldontlie
except Exception as exc:
    print(f"Warning: failed to import balldontlie_client: {exc}")

try:
    from scripts.odds_client import fetch_odds
except Exception as exc:
    print(f"Warning: failed to import odds_client: {exc}")

try:
    from scripts.pitcher_client import fetch_probable_pitchers
except Exception as exc:
    print(f"Warning: failed to import pitcher_client: {exc}")

try:
    from scripts.injury_client import fetch_injuries
except Exception as exc:
    print(f"Warning: failed to import injury_client: {exc}")

try:
    from scripts.bullpen_client import fetch_bullpen_stats
except Exception as exc:
    print(f"Warning: failed to import bullpen_client: {exc}")

try:
    from scripts.platoon_client import fetch_platoon_splits
except Exception as exc:
    print(f"Warning: failed to import platoon_client: {exc}")

try:
    from scripts.umpire_client import fetch_umpire_data
except Exception as exc:
    print(f"Warning: failed to import umpire_client: {exc}")


TEAM_ALIASES = {
    "arizona diamondbacks": "dbacks",
    "d-backs": "dbacks",
    "dbacks": "dbacks",
    "athletics": "athletics",
    "oakland athletics": "athletics",
    "cleveland guardians": "guardians",
    "washington nationals": "nationals",
    "baltimore orioles": "orioles",
    "tampa bay rays": "rays",
    "detroit tigers": "tigers",
    "los angeles angels": "angels",
    "pittsburgh pirates": "pirates",
    "chicago cubs": "cubs",
    "boston red sox": "redsox",
    "atlanta braves": "braves",
    "toronto blue jays": "bluejays",
    "miami marlins": "marlins",
    "new york yankees": "yankees",
    "houston astros": "astros",
    "philadelphia phillies": "phillies",
    "san francisco giants": "giants",
    "chicago white sox": "whitesox",
    "kansas city royals": "royals",
    "seattle mariners": "mariners",
    "minnesota twins": "twins",
    "new york mets": "mets",
    "milwaukee brewers": "brewers",
    "los angeles dodgers": "dodgers",
    "san diego padres": "padres",
    "texas rangers": "rangers",
    "colorado rockies": "rockies",
    "cincinnati reds": "reds",
    "st. louis cardinals": "cardinals",
    "st louis cardinals": "cardinals",
}


def normalize_team_key(value: Any) -> str:
    text = re.sub(r"[^a-z0-9 ]+", "", str(value or "").lower()).strip()
    text = re.sub(r"\s+", " ", text)
    return TEAM_ALIASES.get(text, text.replace(" ", ""))


def frame_to_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        return value.where(pd.notna(value), None).to_dict(orient="records")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def dict_frame_to_records(payload: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    return frame_to_records(payload.get(key))


def attach_schedule_game_ids(
    odds_rows: list[dict[str, Any]],
    schedule_rows: list[dict[str, Any]],
    errors: list[str],
) -> list[dict[str, Any]]:
    """Join aggregated odds to MLB schedule rows so prediction.py can select per game."""
    schedule_index = {}
    for game in schedule_rows:
        pair = (
            normalize_team_key(game.get("home_team")),
            normalize_team_key(game.get("away_team")),
        )
        schedule_index[pair] = game.get("game_id")

    matched = []
    for row in odds_rows:
        pair = (
            normalize_team_key(row.get("home_team")),
            normalize_team_key(row.get("away_team")),
        )
        game_id = schedule_index.get(pair)
        if game_id is None:
            errors.append(
                "Odds matchup not found in MLB schedule: "
                f"{row.get('away_team')} at {row.get('home_team')}"
            )
            continue
        completed = dict(row)
        completed["game_id"] = game_id
        matched.append(completed)
    return matched


class UnifiedSportsModel:
    def __init__(self) -> None:
        self.ball_api_key = (os.getenv("BALLDONTLIE_API_KEY", "") or "").strip()
        self.odds_api_key = (os.getenv("ODDS_API_KEY", "") or "").strip()

    def gather_all_data(self, date_str: str | None = None) -> dict[str, Any]:
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        errors: list[str] = []
        result: dict[str, Any] = {
            "date": date_str,
            "mlb_statsapi": [],
            "savant_statcast": [],
            "retrosheet": [],
            "pybaseball_statcast": [],
            "pybaseball_batting": [],
            "pybaseball_pitching": [],
            "sportsipy_teams": [],
            "sportsipy_player": {},
            "openmeteo_weather": [],
            "balldontlie_teams": [],
            "odds_data": [],
            "pitchers": [],
            "injuries": [],
            "bullpen": [],
            "platoon": [],
            "umpires": [],
            "mlb_team_stats": [],
            "pitcher_data": [],
            "odds": [],
            "weather": [],
            "bullpen_data": [],
            "platoon_data": [],
            "umpire_data": [],
            "errors": errors,
        }

        def safe_call(func: Callable[..., Any] | None, name: str, *args: Any) -> Any:
            if func is None:
                errors.append(f"{name} module not loaded.")
                return {}
            try:
                return func(*args)
            except Exception as exc:
                errors.append(f"{name} fetch error: {exc}")
                return {}

        mlb_stats = safe_call(fetch_mlb_statsapi, "mlb_statsapi", date_str, errors)
        savant = safe_call(fetch_savant_statcast, "savant_statcast", date_str, errors)
        retro = safe_call(fetch_retrosheet, "retrosheet", date_str, errors)
        pyb = safe_call(fetch_pybaseball, "pybaseball", date_str, errors)
        sportsipy = safe_call(fetch_sportsipy, "sportsipy", date_str, errors)
        openmeteo = safe_call(fetch_openmeteo, "openmeteo", date_str, errors)
        balldontlie = safe_call(fetch_balldontlie, "balldontlie", self.ball_api_key, date_str, errors)
        odds_payload = safe_call(fetch_odds, "odds", self.odds_api_key, date_str, errors)
        pitchers = safe_call(fetch_probable_pitchers, "pitchers", date_str, errors)
        injuries = safe_call(fetch_injuries, "injuries", date_str, errors)
        bullpen = safe_call(fetch_bullpen_stats, "bullpen", date_str, errors)
        platoon = safe_call(fetch_platoon_splits, "platoon", 2026, errors)
        umpire = safe_call(fetch_umpire_data, "umpires", date_str, errors)

        result["mlb_statsapi"] = frame_to_records(mlb_stats)
        result["savant_statcast"] = frame_to_records(savant)
        result["retrosheet"] = frame_to_records(retro)
        result["pybaseball_statcast"] = dict_frame_to_records(pyb, "statcast_recent")
        result["pybaseball_batting"] = dict_frame_to_records(pyb, "batting_leaders")
        result["pybaseball_pitching"] = dict_frame_to_records(pyb, "pitching_leaders")
        result["sportsipy_teams"] = dict_frame_to_records(sportsipy, "teams")
        if isinstance(sportsipy, dict) and isinstance(sportsipy.get("player_example"), dict):
            result["sportsipy_player"] = sportsipy["player_example"]
        result["openmeteo_weather"] = frame_to_records(openmeteo)
        result["balldontlie_teams"] = frame_to_records(balldontlie)
        raw_odds_rows = frame_to_records(odds_payload)
        result["odds_data"] = attach_schedule_game_ids(
            raw_odds_rows, result["mlb_statsapi"], errors
        )
        result["pitchers"] = frame_to_records(pitchers)
        result["injuries"] = frame_to_records(injuries)
        result["bullpen"] = frame_to_records(bullpen)
        result["platoon"] = frame_to_records(platoon)
        result["umpires"] = frame_to_records(umpire)

        result["mlb_team_stats"] = result["sportsipy_teams"]
        result["pitcher_data"] = result["pitchers"]
        result["odds"] = result["odds_data"]
        result["weather"] = result["openmeteo_weather"]
        result["bullpen_data"] = result["bullpen"]
        result["platoon_data"] = result["platoon"]
        result["umpire_data"] = result["umpires"]

        report_directory = Path("report")
        if report_directory.exists() and report_directory.is_file():
            report_directory.unlink()
        report_directory.mkdir(parents=True, exist_ok=True)
        (report_directory / f"{date_str}.json").write_text(
            json.dumps(result, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

        schedule_error_messages = [
            str(message)
            for message in errors
            if ("mlb_statsapi" in str(message).lower() or "mlb stats api" in str(message).lower())
            and (
                "fetch error" in str(message).lower()
                or "module not loaded" in str(message).lower()
                or "api error" in str(message).lower()
            )
        ]
        if not result["mlb_statsapi"] and schedule_error_messages:
            raise RuntimeError("schedule fetch failed: " + "; ".join(schedule_error_messages))

        return result
