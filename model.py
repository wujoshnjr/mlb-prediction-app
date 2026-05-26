# model.py
"""Unified external data collection layer for MLB prediction generation.

This module gathers available upstream data sources while keeping optional
providers defensive. It also exposes backward-compatible aliases consumed by
prediction.py.

Runtime messages and comments are ASCII-only to reduce encoding risk during
browser-based edits.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# Optional providers. Import failures must not prevent the application from
# starting, but the primary MLB schedule provider is validated during runtime.
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


def frame_to_records(value: Any) -> list[dict[str, Any]]:
    """Convert a DataFrame or list payload into serializable record objects."""
    if value is None:
        return []

    if isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        return value.to_dict(orient="records")

    if isinstance(value, list):
        return [
            item for item in value
            if isinstance(item, dict)
        ]

    return []


def dict_frame_to_records(
    payload: Any,
    key: str,
) -> list[dict[str, Any]]:
    """Read a DataFrame/list item from a dictionary payload safely."""
    if not isinstance(payload, dict):
        return []

    return frame_to_records(payload.get(key))


class UnifiedSportsModel:
    """Gather all currently available upstream MLB data sources."""

    def __init__(self) -> None:
        raw_ball = os.getenv("BALLDONTLIE_API_KEY", "") or ""
        raw_odds = os.getenv("ODDS_API_KEY", "") or ""

        self.ball_api_key = (
            raw_ball.strip()
            .replace("\n", "")
            .replace("\r", "")
        )
        self.odds_api_key = (
            raw_odds.strip()
            .replace("\n", "")
            .replace("\r", "")
        )

    def gather_all_data(self, date_str: str | None = None) -> dict[str, Any]:
        """Gather provider payloads and expose the prediction.py contract."""
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
            # Aliases currently consumed by prediction.py.
            "mlb_team_stats": [],
            "pitcher_data": [],
            "odds": [],
            "weather": [],
            "bullpen_data": [],
            "platoon_data": [],
            "umpire_data": [],
            "errors": errors,
        }

        def safe_call(
            func: Callable[..., Any] | None,
            name: str,
            *args: Any,
        ) -> Any:
            """Run an optional provider without hiding failures."""
            if func is None:
                errors.append(f"{name} module not loaded.")
                return {}

            try:
                return func(*args)
            except Exception as exc:
                errors.append(f"{name} fetch error: {exc}")
                return {}

        mlb_stats = safe_call(
            fetch_mlb_statsapi,
            "mlb_statsapi",
            date_str,
            errors,
        )
        savant = safe_call(
            fetch_savant_statcast,
            "savant_statcast",
            date_str,
            errors,
        )
        retro = safe_call(
            fetch_retrosheet,
            "retrosheet",
            date_str,
            errors,
        )
        pyb = safe_call(
            fetch_pybaseball,
            "pybaseball",
            date_str,
            errors,
        )
        sportsipy = safe_call(
            fetch_sportsipy,
            "sportsipy",
            date_str,
            errors,
        )
        openmeteo = safe_call(
            fetch_openmeteo,
            "openmeteo",
            date_str,
            errors,
        )
        balldontlie = safe_call(
            fetch_balldontlie,
            "balldontlie",
            self.ball_api_key,
            date_str,
            errors,
        )
        odds_payload = safe_call(
            fetch_odds,
            "odds",
            self.odds_api_key,
            date_str,
            errors,
        )
        pitchers = safe_call(
            fetch_probable_pitchers,
            "pitchers",
            date_str,
            errors,
        )
        injuries = safe_call(
            fetch_injuries,
            "injuries",
            date_str,
            errors,
        )
        bullpen = safe_call(
            fetch_bullpen_stats,
            "bullpen",
            date_str,
            errors,
        )
        platoon = safe_call(
            fetch_platoon_splits,
            "platoon",
            2026,
            errors,
        )
        umpire = safe_call(
            fetch_umpire_data,
            "umpires",
            date_str,
            errors,
        )

        result["mlb_statsapi"] = frame_to_records(mlb_stats)
        result["savant_statcast"] = frame_to_records(savant)
        result["retrosheet"] = frame_to_records(retro)

        result["pybaseball_statcast"] = dict_frame_to_records(
            pyb,
            "statcast_recent",
        )
        result["pybaseball_batting"] = dict_frame_to_records(
            pyb,
            "batting_leaders",
        )
        result["pybaseball_pitching"] = dict_frame_to_records(
            pyb,
            "pitching_leaders",
        )

        result["sportsipy_teams"] = dict_frame_to_records(
            sportsipy,
            "teams",
        )
        if isinstance(sportsipy, dict):
            player_example = sportsipy.get("player_example", {})
            if isinstance(player_example, dict):
                result["sportsipy_player"] = player_example

        result["openmeteo_weather"] = frame_to_records(openmeteo)
        result["balldontlie_teams"] = frame_to_records(balldontlie)
        result["odds_data"] = frame_to_records(odds_payload)
        result["pitchers"] = frame_to_records(pitchers)
        result["injuries"] = frame_to_records(injuries)
        result["bullpen"] = frame_to_records(bullpen)
        result["platoon"] = frame_to_records(platoon)
        result["umpires"] = frame_to_records(umpire)

        # Backward-compatible aliases required by prediction.py.
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

        report_path = report_directory / f"{date_str}.json"
        report_path.write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )

        # A real MLB off-day may legitimately have an empty schedule.
        # A provider/import/request failure must not be published as an off-day.
        schedule_error_messages = [
            str(message)
            for message in errors
            if (
                "mlb_statsapi" in str(message).lower()
                or "mlb stats api" in str(message).lower()
            )
            and (
                "fetch error" in str(message).lower()
                or "module not loaded" in str(message).lower()
                or "api error" in str(message).lower()
            )
        ]

        if not result["mlb_statsapi"] and schedule_error_messages:
            raise RuntimeError(
                "schedule fetch failed: "
                + "; ".join(schedule_error_messages)
            )

        return result
