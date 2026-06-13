"""Unified upstream data collection with a stable prediction.py contract."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
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
    "az diamondbacks": "dbacks",
    "ari diamondbacks": "dbacks",
    "d-backs": "dbacks",
    "dbacks": "dbacks",

    "athletics": "athletics",
    "oakland athletics": "athletics",
    "oakland a's": "athletics",
    "oakland as": "athletics",

    "cleveland guardians": "guardians",

    "washington nationals": "nationals",
    "wsh nationals": "nationals",

    "baltimore orioles": "orioles",

    "tampa bay rays": "rays",
    "tb rays": "rays",
    "tbr rays": "rays",

    "detroit tigers": "tigers",

    "los angeles angels": "angels",
    "la angels": "angels",
    "laa angels": "angels",

    "pittsburgh pirates": "pirates",

    "chicago cubs": "cubs",
    "chi cubs": "cubs",

    "boston red sox": "redsox",

    "atlanta braves": "braves",

    "toronto blue jays": "bluejays",

    "miami marlins": "marlins",

    "new york yankees": "yankees",
    "ny yankees": "yankees",
    "nyy": "yankees",

    "houston astros": "astros",

    "philadelphia phillies": "phillies",

    "san francisco giants": "giants",
    "sf giants": "giants",
    "sfg giants": "giants",

    "chicago white sox": "whitesox",
    "chi white sox": "whitesox",
    "cws white sox": "whitesox",

    "kansas city royals": "royals",
    "kc royals": "royals",
    "kcr royals": "royals",

    "seattle mariners": "mariners",

    "minnesota twins": "twins",

    "new york mets": "mets",
    "ny mets": "mets",
    "nym": "mets",

    "milwaukee brewers": "brewers",

    "los angeles dodgers": "dodgers",
    "la dodgers": "dodgers",
    "lad dodgers": "dodgers",

    "san diego padres": "padres",
    "sd padres": "padres",
    "sdp padres": "padres",

    "texas rangers": "rangers",

    "colorado rockies": "rockies",

    "cincinnati reds": "reds",

    "st. louis cardinals": "cardinals",
    "st louis cardinals": "cardinals",
    "stl cardinals": "cardinals",
}


def normalize_team_key(value: Any) -> str:
    text = re.sub(r"[^a-z0-9 ]+", "", str(value or "").lower()).strip()
    text = re.sub(r"\s+", " ", text)
    return TEAM_ALIASES.get(text, text.replace(" ", ""))


def _parse_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        parsed = datetime.fromisoformat(text)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    except Exception:
        try:
            parsed_ts = pd.to_datetime(value, utc=True, errors="coerce")
            if pd.isna(parsed_ts):
                return None

            return parsed_ts.to_pydatetime()

        except Exception:
            return None


def _time_delta_hours(a: Any, b: Any) -> float | None:
    left = _parse_utc_datetime(a)
    right = _parse_utc_datetime(b)

    if left is None or right is None:
        return None

    return abs((left - right).total_seconds()) / 3600.0


def _schedule_time_value(game: dict[str, Any]) -> Any:
    for key in (
        "start_time",
        "game_datetime",
        "game_date",
        "commence_time",
        "official_date",
    ):
        value = game.get(key)
        if value:
            return value

    return None


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
    """Join aggregated odds to MLB schedule rows so prediction.py can select per game.

    Matching layers:
    1. direct home/away pair
    2. swapped pair
    3. team-set fallback
    4. team + time-window fallback

    The odds row's original home/away odds direction is preserved.
    Only schedule game_id and audit fields are attached.
    """
    if not odds_rows:
        return []

    if not schedule_rows:
        errors.append("No MLB schedule rows available for odds matching.")
        return []

    schedule_direct: dict[tuple[str, str], dict[str, Any]] = {}
    schedule_team_sets: dict[frozenset[str], list[dict[str, Any]]] = {}

    for game in schedule_rows:
        home_key = normalize_team_key(game.get("home_team"))
        away_key = normalize_team_key(game.get("away_team"))

        if not home_key or not away_key:
            continue

        pair = (home_key, away_key)
        team_set = frozenset({home_key, away_key})

        schedule_direct[pair] = game
        schedule_team_sets.setdefault(team_set, []).append(game)

    matched: list[dict[str, Any]] = []

    for row in odds_rows:
        home_key = normalize_team_key(row.get("home_team"))
        away_key = normalize_team_key(row.get("away_team"))

        pair = (home_key, away_key)
        swapped_pair = (away_key, home_key)
        team_set = frozenset({home_key, away_key})

        match_game: dict[str, Any] | None = None
        match_type = ""
        match_confidence = 0.0

        if pair in schedule_direct:
            match_game = schedule_direct[pair]
            match_type = "direct_pair"
            match_confidence = 1.0

        elif swapped_pair in schedule_direct:
            match_game = schedule_direct[swapped_pair]
            match_type = "swapped_pair"
            match_confidence = 0.8

        else:
            candidates = schedule_team_sets.get(team_set, [])

            if len(candidates) == 1:
                match_game = candidates[0]
                match_type = "team_set"
                match_confidence = 0.7

            elif candidates:
                row_time = row.get("commence_time")
                best_candidate: dict[str, Any] | None = None
                best_delta: float | None = None

                for candidate in candidates:
                    delta = _time_delta_hours(
                        row_time,
                        _schedule_time_value(candidate),
                    )

                    if delta is None:
                        continue

                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        best_candidate = candidate

                if (
                    best_candidate is not None
                    and best_delta is not None
                    and best_delta <= 6
                ):
                    match_game = best_candidate
                    match_type = "team_time_window"
                    match_confidence = 0.6

        if match_game is None:
            errors.append(
                "Odds matchup not found in MLB schedule: "
                f"{row.get('away_team')} at {row.get('home_team')}; "
                f"normalized_pair=({away_key}, {home_key})"
            )
            continue

        completed = dict(row)
        completed["game_id"] = match_game.get("game_id")
        completed["odds_schedule_match_type"] = match_type
        completed["odds_schedule_match_confidence"] = match_confidence
        completed["odds_provider_event_id"] = row.get("event_id")
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
