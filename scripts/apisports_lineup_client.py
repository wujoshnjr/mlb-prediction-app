from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

BASE_URL = "https://v1.baseball.api-sports.io"
ENV_API_KEY = "APISPORTS_BASEBALL_KEY"
ENV_LEAGUE_ID = "APISPORTS_BASEBALL_LEAGUE_ID"
DEFAULT_LEAGUE_ID = "1"
LINEUP_SOURCE = "apisports_baseball"

OUTPUT_COLUMNS = [
    "game_id",
    "game_date",
    "api_sports_game_id",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "home_lineup_confirmed",
    "away_lineup_confirmed",
    "home_lineup_player_ids_json",
    "away_lineup_player_ids_json",
    "home_lineup_player_count",
    "away_lineup_player_count",
    "home_top3_player_ids",
    "away_top3_player_ids",
    "lineup_source",
    "lineup_fetched_at",
    "lineup_error",
]


def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
    """Safely traverse nested dict/list objects using dot notation."""
    try:
        current = obj
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                index = int(part)
                if index < 0 or index >= len(current):
                    return default
                current = current[index]
            else:
                return default

            if current is None:
                return default

        return current
    except Exception:
        return default


def _safe_int(value: Any) -> Optional[int]:
    """Convert value to int without raising."""
    try:
        if value is None:
            return None

        if isinstance(value, bool):
            return int(value)

        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None

        return int(number)
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str:
    """Return clean string. Missing/null-like values become empty string."""
    if value is None:
        return ""

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""

    return text


def _current_utc_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _empty_frame() -> pd.DataFrame:
    """Return empty lineup DataFrame with standard columns."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def load_api_key() -> str:
    """Load API-Sports key from environment variable."""
    return _clean_text(os.environ.get(ENV_API_KEY))


def _api_get(
    path: str,
    params: Dict[str, Any],
    api_key: str,
    timeout: int,
    errors: Optional[List[str]] = None,
    noisy_404: bool = False,
) -> Optional[Dict[str, Any]]:
    """Safely perform API-Sports GET request and return parsed JSON."""
    url = f"{BASE_URL}{path}"
    headers = {
        "x-apisports-key": api_key,
        "Accept": "application/json",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        if errors is not None:
            errors.append(f"Request failed for {path}: {exc}")
        return None

    if response.status_code != 200:
        if errors is not None and (response.status_code != 404 or noisy_404):
            errors.append(
                f"API error {response.status_code} for {path}: "
                f"{response.text[:200]}"
            )
        return None

    try:
        payload = response.json()
    except ValueError as exc:
        if errors is not None:
            errors.append(f"JSON decode error for {path}: {exc}")
        return None

    if not isinstance(payload, dict):
        if errors is not None:
            errors.append(f"Non-dict JSON response for {path}")
        return None

    return payload


def _extract_first_int(obj: Any, paths: List[str]) -> Optional[int]:
    """Try multiple paths and return first valid integer."""
    for path in paths:
        value = _safe_get(obj, path)
        result = _safe_int(value)
        if result is not None:
            return result
    return None


def _extract_first_text(obj: Any, paths: List[str]) -> str:
    """Try multiple paths and return first clean text."""
    for path in paths:
        value = _clean_text(_safe_get(obj, path))
        if value:
            return value
    return ""


def _dedupe_ids(values: List[int]) -> List[int]:
    """Deduplicate integer IDs while preserving order."""
    seen = set()
    result: List[int] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result


def _extract_player_id(item: Any) -> Optional[int]:
    """Extract player ID from common API-Sports lineup/player shapes."""
    if isinstance(item, (int, float, str)):
        return _safe_int(item)

    if not isinstance(item, dict):
        return None

    return _extract_first_int(
        item,
        [
            "player.id",
            "player.player_id",
            "player.playerId",
            "athlete.id",
            "person.id",
            "id",
            "player_id",
            "playerId",
        ],
    )


def _extract_player_ids_from_any(value: Any) -> List[int]:
    """Extract player IDs from arbitrary lineup/list/dict payload."""
    player_ids: List[int] = []

    if isinstance(value, list):
        for item in value:
            player_id = _extract_player_id(item)
            if player_id is not None:
                player_ids.append(player_id)
            elif isinstance(item, (dict, list)):
                player_ids.extend(_extract_player_ids_from_any(item))

    elif isinstance(value, dict):
        direct_id = _extract_player_id(value)
        if direct_id is not None:
            player_ids.append(direct_id)

        for key in (
            "lineup",
            "players",
            "batters",
            "starting",
            "starters",
            "roster",
            "athletes",
        ):
            nested = value.get(key)
            if nested is not None:
                player_ids.extend(_extract_player_ids_from_any(nested))

    return _dedupe_ids(player_ids)


def _team_id_matches(container: Any, target_team_id: Optional[int]) -> bool:
    """Return True when container appears to refer to target team id."""
    if target_team_id is None or not isinstance(container, dict):
        return False

    possible_team_id = _extract_first_int(
        container,
        [
            "team.id",
            "team_id",
            "teamId",
            "id",
        ],
    )

    return possible_team_id == target_team_id


def _search_lineup_for_team(payload: Any, target_team_id: Optional[int]) -> List[int]:
    """Recursively search payload for lineup belonging to target team."""
    if target_team_id is None:
        return []

    visited = set()

    def walk(obj: Any) -> List[int]:
        if isinstance(obj, dict):
            object_id = id(obj)
            if object_id in visited:
                return []
            visited.add(object_id)

            if _team_id_matches(obj, target_team_id):
                for key in (
                    "lineup",
                    "players",
                    "batters",
                    "starting",
                    "starters",
                    "roster",
                    "athletes",
                ):
                    if key in obj:
                        ids = _extract_player_ids_from_any(obj.get(key))
                        if ids:
                            return ids

            for value in obj.values():
                result = walk(value)
                if result:
                    return result

        elif isinstance(obj, list):
            for item in obj:
                result = walk(item)
                if result:
                    return result

        return []

    return walk(payload)


def _search_lineup_by_side(payload: Any, side: str) -> List[int]:
    """Search common home/away side paths for lineup player IDs."""
    side = side.lower().strip()
    if side not in {"home", "away"}:
        return []

    candidate_paths = [
        f"lineups.{side}",
        f"lineups.{side}.players",
        f"teams.{side}.lineup",
        f"teams.{side}.players",
        f"{side}.lineup",
        f"{side}.players",
        f"response.0.lineups.{side}",
        f"response.0.lineups.{side}.players",
        f"response.0.teams.{side}.lineup",
        f"response.0.teams.{side}.players",
    ]

    for path in candidate_paths:
        candidate = _safe_get(payload, path)
        ids = _extract_player_ids_from_any(candidate)
        if ids:
            return ids

    return []


def parse_lineup_players(payload: Any, side: str) -> List[int]:
    """Parse home or away lineup player IDs from a defensive API payload."""
    return _search_lineup_by_side(payload, side)


def _extract_lineup_home_away(
    payload: Any,
    home_team_id_api: Optional[int],
    away_team_id_api: Optional[int],
) -> Tuple[List[int], List[int]]:
    """Extract home and away lineups from an endpoint response."""
    home_ids = parse_lineup_players(payload, "home")
    away_ids = parse_lineup_players(payload, "away")

    if not home_ids:
        home_ids = _search_lineup_for_team(payload, home_team_id_api)
    if not away_ids:
        away_ids = _search_lineup_for_team(payload, away_team_id_api)

    return _dedupe_ids(home_ids), _dedupe_ids(away_ids)


def _extract_api_game_id(game: Dict[str, Any]) -> Optional[int]:
    """Extract API-Sports game id from common response shapes."""
    return _extract_first_int(
        game,
        [
            "id",
            "game.id",
            "fixture.id",
        ],
    )


def _extract_mlb_game_id(game: Dict[str, Any]) -> Optional[int]:
    """Try to extract MLB Stats API game id from API-Sports game payload."""
    return _extract_first_int(
        game,
        [
            "idMLB",
            "id_mlb",
            "mlb_id",
            "mlbId",
            "external_id",
            "externalId",
            "game.idMLB",
            "game.id_mlb",
            "game.mlb_id",
            "game.mlbId",
            "game.external_id",
            "game.externalId",
            "fixture.idMLB",
            "fixture.id_mlb",
            "fixture.mlb_id",
            "fixture.mlbId",
            "fixture.external_id",
            "fixture.externalId",
        ],
    )


def _extract_team_info(game: Dict[str, Any], side: str) -> Dict[str, Any]:
    """Extract team id/name for side from common response shapes."""
    side = side.lower().strip()
    team_obj = (
        _safe_get(game, f"teams.{side}", {})
        or _safe_get(game, f"team.{side}", {})
        or _safe_get(game, f"{side}.team", {})
        or _safe_get(game, side, {})
    )

    if not isinstance(team_obj, dict):
        team_obj = {}

    api_team_id = _extract_first_int(
        team_obj,
        [
            "id",
            "team.id",
        ],
    )

    mlb_team_id = _extract_first_int(
        team_obj,
        [
            "idMLB",
            "id_mlb",
            "mlb_id",
            "mlbId",
            "team.idMLB",
            "team.id_mlb",
            "team.mlb_id",
            "team.mlbId",
        ],
    )

    name = _extract_first_text(
        team_obj,
        [
            "name",
            "team.name",
        ],
    )

    return {
        "api_team_id": api_team_id,
        "mlb_team_id": mlb_team_id,
        "final_team_id": mlb_team_id if mlb_team_id is not None else api_team_id,
        "name": name,
    }


def _build_row(
    *,
    date_str: str,
    game: Dict[str, Any],
    api_game_id: int,
    home_ids: List[int],
    away_ids: List[int],
    game_errors: List[str],
) -> Dict[str, Any]:
    """Build one output lineup row."""
    home_team = _extract_team_info(game, "home")
    away_team = _extract_team_info(game, "away")

    mlb_game_id = _extract_mlb_game_id(game)
    effective_game_id = str(mlb_game_id) if mlb_game_id is not None else str(api_game_id)

    home_ids = _dedupe_ids(home_ids)
    away_ids = _dedupe_ids(away_ids)

    return {
        "game_id": effective_game_id,
        "game_date": date_str,
        "api_sports_game_id": api_game_id,
        "home_team_id": home_team["final_team_id"] or "",
        "away_team_id": away_team["final_team_id"] or "",
        "home_team": home_team["name"],
        "away_team": away_team["name"],
        "home_lineup_confirmed": len(home_ids) >= 9,
        "away_lineup_confirmed": len(away_ids) >= 9,
        "home_lineup_player_ids_json": json.dumps(home_ids, ensure_ascii=True),
        "away_lineup_player_ids_json": json.dumps(away_ids, ensure_ascii=True),
        "home_lineup_player_count": len(home_ids),
        "away_lineup_player_count": len(away_ids),
        "home_top3_player_ids": ",".join(str(player_id) for player_id in home_ids[:3]),
        "away_top3_player_ids": ",".join(str(player_id) for player_id in away_ids[:3]),
        "lineup_source": LINEUP_SOURCE,
        "lineup_fetched_at": _current_utc_iso(),
        "lineup_error": "; ".join(message for message in game_errors if message),
    }


def fetch_apisports_lineups(
    date_str: Optional[str] = None,
    errors: Optional[List[str]] = None,
    timeout: int = 20,
    sleep_seconds: float = 0.1,
) -> pd.DataFrame:
    """Fetch MLB lineup fallback rows from API-Sports Baseball API."""
    if errors is None:
        errors = []

    api_key = load_api_key()
    if not api_key:
        errors.append("APISPORTS_BASEBALL_KEY missing")
        return _empty_frame()

    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        season = str(datetime.strptime(date_str, "%Y-%m-%d").year)
    except ValueError:
        errors.append(f"Invalid date format: {date_str}")
        return _empty_frame()

    league_id = _clean_text(os.environ.get(ENV_LEAGUE_ID)) or DEFAULT_LEAGUE_ID

    schedule_payload = _api_get(
        "/games",
        {
            "date": date_str,
            "league": league_id,
            "season": season,
        },
        api_key,
        timeout,
        errors=errors,
        noisy_404=True,
    )

    if schedule_payload is None:
        return _empty_frame()

    games = schedule_payload.get("response", [])
    if not isinstance(games, list) or not games:
        return _empty_frame()

    rows: List[Dict[str, Any]] = []

    for game in games:
        if not isinstance(game, dict):
            continue

        api_game_id = _extract_api_game_id(game)
        if api_game_id is None:
            continue

        home_team = _extract_team_info(game, "home")
        away_team = _extract_team_info(game, "away")

        home_team_id_api = _safe_int(home_team.get("api_team_id"))
        away_team_id_api = _safe_int(away_team.get("api_team_id"))

        game_errors: List[str] = []
        home_ids: List[int] = []
        away_ids: List[int] = []

        for endpoint in ("/games/lineups", "/games/statistics", "/games/events"):
            payload = _api_get(
                endpoint,
                {"game": api_game_id},
                api_key,
                timeout,
                errors=None,
                noisy_404=False,
            )

            if payload is None:
                continue

            candidate_home, candidate_away = _extract_lineup_home_away(
                payload,
                home_team_id_api,
                away_team_id_api,
            )

            if len(candidate_home) > len(home_ids):
                home_ids = candidate_home
            if len(candidate_away) > len(away_ids):
                away_ids = candidate_away

            if len(home_ids) >= 9 and len(away_ids) >= 9:
                break

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if len(home_ids) < 9 and len(away_ids) < 9:
            game_errors.append("No usable confirmed lineup found from API-Sports endpoints")

        rows.append(
            _build_row(
                date_str=date_str,
                game=game,
                api_game_id=api_game_id,
                home_ids=home_ids,
                away_ids=away_ids,
                game_errors=game_errors,
            )
        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    frame = frame[OUTPUT_COLUMNS].fillna("")

    return frame


if __name__ == "__main__":
    cli_errors: List[str] = []
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = fetch_apisports_lineups(
        date_str=today_utc,
        errors=cli_errors,
        timeout=20,
        sleep_seconds=0.15,
    )

    summary = {
        "date": today_utc,
        "rows": int(len(df)),
        "home_confirmed_rows": int(df["home_lineup_confirmed"].sum()) if not df.empty else 0,
        "away_confirmed_rows": int(df["away_lineup_confirmed"].sum()) if not df.empty else 0,
        "errors": cli_errors,
    }

    print(json.dumps(summary, indent=2, default=str))
    if not df.empty:
        print(df.head().to_string(index=False))
