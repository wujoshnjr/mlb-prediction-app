from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_LIVE_GAME_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"

REQUEST_TIMEOUT = 20
LINEUP_SOURCE = "mlb_statsapi_live_data"

LINEUP_COLUMNS = [
    "game_id",
    "game_date",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "home_lineup_confirmed",
    "away_lineup_confirmed",
    "home_lineup_player_ids_json",
    "away_lineup_player_ids_json",
    "home_lineup_player_names_json",
    "away_lineup_player_names_json",
    "lineup_source",
    "lineup_fetched_at",
]


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO string ending in Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _log_error(errors: Optional[List[str]], message: str) -> None:
    """Append an error message when the caller provided an error list."""
    if errors is not None:
        errors.append(message)


def _fetch_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
    label: str = "request",
) -> Optional[Dict[str, Any]]:
    """Fetch one MLB Stats API JSON payload without raising to the caller."""
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload

        _log_error(errors, f"{label} returned a non-object JSON payload.")
        return None
    except Exception as exc:
        _log_error(errors, f"{label} failed: {exc}")
        return None


def _normalise_player_id(value: Any) -> Optional[int]:
    """Convert a player identifier into an integer, or return None."""
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _player_name_from_boxscore(
    player_id: int,
    players: Dict[str, Any],
) -> str:
    """Read a player's full name from a boxscore players mapping."""
    player_payload = players.get(f"ID{player_id}", {})
    if not isinstance(player_payload, dict):
        return ""

    person = player_payload.get("person", {})
    if not isinstance(person, dict):
        return ""

    return str(person.get("fullName") or "")


def _extract_team_lineup(
    boxscore_team: Dict[str, Any],
) -> tuple[bool, List[int], List[str]]:
    """Extract an official starting batting order from one team boxscore.

    MLB live payloads may expose battingOrder as player ids or as objects.
    A lineup is treated as confirmed only when at least nine unique batter
    ids can be obtained. Only the first nine lineup positions are stored.
    """
    if not isinstance(boxscore_team, dict):
        return False, [], []

    batting_order = boxscore_team.get("battingOrder", [])
    players = boxscore_team.get("players", {})

    if not isinstance(batting_order, list) or len(batting_order) < 9:
        return False, [], []

    if not isinstance(players, dict):
        players = {}

    player_ids: List[int] = []
    player_names: List[str] = []
    seen_ids: set[int] = set()

    for entry in batting_order:
        player_id: Optional[int] = None
        player_name = ""

        if isinstance(entry, dict):
            player_id = _normalise_player_id(
                entry.get("id") or entry.get("personId")
            )
            player_name = str(
                entry.get("fullName")
                or entry.get("name")
                or ""
            )
        else:
            player_id = _normalise_player_id(entry)

        if player_id is None or player_id in seen_ids:
            continue

        if not player_name:
            player_name = _player_name_from_boxscore(
                player_id,
                players,
            )

        seen_ids.add(player_id)
        player_ids.append(player_id)
        player_names.append(player_name)

        if len(player_ids) == 9:
            break

    if len(player_ids) < 9:
        return False, [], []

    return True, player_ids, player_names


def _empty_lineup_json() -> str:
    """Return a canonical empty JSON array string."""
    return json.dumps([], ensure_ascii=True)


def fetch_confirmed_lineups(
    date_str: Optional[str] = None,
    errors: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Retrieve confirmed MLB starting lineups for games on one date.

    One row is returned for every scheduled game retrieved successfully.
    A lineup is marked confirmed only when an actual batting order containing
    at least nine unique players is present in the MLB live game payload.
    Probable or expected lineups are never treated as confirmed.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    fetched_at = _utc_now_iso()

    schedule_payload = _fetch_json(
        MLB_SCHEDULE_URL,
        params={
            "sportId": 1,
            "date": date_str,
            "gameTypes": "R",
        },
        errors=errors,
        label="MLB lineup schedule fetch",
    )

    if schedule_payload is None:
        return pd.DataFrame(columns=LINEUP_COLUMNS)

    games: List[Dict[str, Any]] = []
    for date_payload in schedule_payload.get("dates", []) or []:
        if not isinstance(date_payload, dict):
            continue
        for game in date_payload.get("games", []) or []:
            if isinstance(game, dict):
                games.append(game)

    if not games:
        return pd.DataFrame(columns=LINEUP_COLUMNS)

    rows: List[Dict[str, Any]] = []

    for game in games:
        game_id = game.get("gamePk")
        if game_id is None:
            _log_error(
                errors,
                "MLB lineup schedule row missing gamePk; row skipped.",
            )
            continue

        teams = game.get("teams", {})
        if not isinstance(teams, dict):
            teams = {}

        home = teams.get("home", {})
        away = teams.get("away", {})
        if not isinstance(home, dict):
            home = {}
        if not isinstance(away, dict):
            away = {}

        home_team_payload = home.get("team", {})
        away_team_payload = away.get("team", {})
        if not isinstance(home_team_payload, dict):
            home_team_payload = {}
        if not isinstance(away_team_payload, dict):
            away_team_payload = {}

        home_team_id = home_team_payload.get("id")
        away_team_id = away_team_payload.get("id")
        home_team = str(home_team_payload.get("name") or "")
        away_team = str(away_team_payload.get("name") or "")

        home_confirmed = False
        away_confirmed = False
        home_player_ids: List[int] = []
        away_player_ids: List[int] = []
        home_player_names: List[str] = []
        away_player_names: List[str] = []

        live_payload = _fetch_json(
            MLB_LIVE_GAME_URL.format(game_id=game_id),
            errors=errors,
            label=f"MLB lineup live feed fetch for game {game_id}",
        )

        if live_payload is not None:
            try:
                live_data = live_payload.get("liveData", {})
                if not isinstance(live_data, dict):
                    live_data = {}

                boxscore = live_data.get("boxscore", {})
                if not isinstance(boxscore, dict):
                    boxscore = {}

                boxscore_teams = boxscore.get("teams", {})
                if not isinstance(boxscore_teams, dict):
                    boxscore_teams = {}

                home_boxscore = boxscore_teams.get("home", {})
                away_boxscore = boxscore_teams.get("away", {})

                (
                    home_confirmed,
                    home_player_ids,
                    home_player_names,
                ) = _extract_team_lineup(home_boxscore)

                (
                    away_confirmed,
                    away_player_ids,
                    away_player_names,
                ) = _extract_team_lineup(away_boxscore)

            except Exception as exc:
                _log_error(
                    errors,
                    f"MLB lineup parse failed for game {game_id}: {exc}",
                )

        rows.append(
            {
                "game_id": game_id,
                "game_date": date_str,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team": home_team,
                "away_team": away_team,
                "home_lineup_confirmed": home_confirmed,
                "away_lineup_confirmed": away_confirmed,
                "home_lineup_player_ids_json": (
                    json.dumps(home_player_ids, ensure_ascii=True)
                    if home_confirmed
                    else _empty_lineup_json()
                ),
                "away_lineup_player_ids_json": (
                    json.dumps(away_player_ids, ensure_ascii=True)
                    if away_confirmed
                    else _empty_lineup_json()
                ),
                "home_lineup_player_names_json": (
                    json.dumps(home_player_names, ensure_ascii=True)
                    if home_confirmed
                    else _empty_lineup_json()
                ),
                "away_lineup_player_names_json": (
                    json.dumps(away_player_names, ensure_ascii=True)
                    if away_confirmed
                    else _empty_lineup_json()
                ),
                "lineup_source": LINEUP_SOURCE,
                "lineup_fetched_at": fetched_at,
            }
        )

    return pd.DataFrame(rows, columns=LINEUP_COLUMNS)
