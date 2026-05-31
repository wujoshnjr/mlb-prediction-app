from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

# ---------------------------------------------------------------------------

# MLB Stats API endpoints

# ---------------------------------------------------------------------------

MLB_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
MLB_BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"

# ---------------------------------------------------------------------------

# Helper utilities

# ---------------------------------------------------------------------------

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

```
        if current is None:
            return default

    return current
except Exception:
    return default
```

def _safe_int(value: Any) -> Optional[int]:
"""Convert a value to int without raising. Non-finite values return None."""
try:
if value is None:
return None

```
    if isinstance(value, bool):
        return int(value)

    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None

    return int(number)
except (TypeError, ValueError):
    return None
```

def _safe_float(value: Any) -> Optional[float]:
"""Convert a value to float without raising. Non-finite values return None."""
try:
if value is None:
return None

```
    if isinstance(value, bool):
        return float(value)

    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None

    return float(number)
except (TypeError, ValueError):
    return None
```

def _safe_str(value: Any) -> str:
"""Return a clean string. None becomes an empty string."""
if value is None:
return ""

```
text = str(value)
if text.lower() in {"nan", "none", "null"}:
    return ""

return text
```

def _parse_wind(wind_text: Optional[str]) -> Tuple[Optional[float], str]:
"""Parse an MLB wind string into speed and direction.

```
Examples:
- "7 mph, Out To CF" -> (7.0, "Out To CF")
- "12 mph" -> (12.0, "12 mph")
"""
if not isinstance(wind_text, str) or not wind_text.strip():
    return None, ""

text = wind_text.strip()

match = re.match(r"(\d+(?:\.\d+)?)\s*mph,\s*(.*)", text, re.IGNORECASE)
if match:
    return _safe_float(match.group(1)), match.group(2).strip()

match = re.match(r"(\d+(?:\.\d+)?)\s*mph", text, re.IGNORECASE)
if match:
    return _safe_float(match.group(1)), text

return None, text
```

def _player_key(player_id: Any) -> str:
"""Return MLB boxscore players-map key, e.g. 680694 -> ID680694."""
player_id_int = _safe_int(player_id)
if player_id_int is not None:
return f"ID{player_id_int}"

```
return f"ID{str(player_id).strip()}"
```

def _get_player_from_players_map(players_map: Any, player_id: Any) -> Dict[str, Any]:
"""Look up player object in MLB boxscore players map."""
if not isinstance(players_map, dict):
return {}

```
player = players_map.get(_player_key(player_id))
if isinstance(player, dict):
    return player

return {}
```

def _extract_player_name(player_obj: Any) -> str:
"""Extract a player name from several MLB Stats API object shapes."""
if not isinstance(player_obj, dict):
return ""

```
for path in (
    "person.fullName",
    "fullName",
    "player.fullName",
    "player.person.fullName",
):
    name = _safe_str(_safe_get(player_obj, path))
    if name:
        return name

player_id = (
    _safe_get(player_obj, "person.id")
    or _safe_get(player_obj, "id")
    or _safe_get(player_obj, "player.id")
)
if player_id is not None:
    return str(player_id)

return ""
```

def _extract_player_ids_from_entries(entries: Any) -> List[int]:
"""Extract player IDs from MLB battingOrder / batters / pitchers entries.

```
Supported entry shapes:
- [680694, 123456]
- ["680694", "123456"]
- [{"id": 680694}, {"person": {"id": 123456}}]
- [{"player": {"id": 680694}}]
"""
ids: List[int] = []

if not isinstance(entries, list):
    return ids

for entry in entries:
    player_id: Optional[int] = None

    if isinstance(entry, (int, float, str)):
        player_id = _safe_int(entry)
    elif isinstance(entry, dict):
        player_id = (
            _safe_int(entry.get("id"))
            or _safe_int(entry.get("playerID"))
            or _safe_int(_safe_get(entry, "person.id"))
            or _safe_int(_safe_get(entry, "player.id"))
        )

    if player_id is not None:
        ids.append(player_id)

return ids
```

def _game_has_started_or_finished(game_status: str, detailed_state: str) -> bool:
"""Conservative live/final detector for starter confirmation."""
status_upper = _safe_str(game_status).upper()
detailed_lower = _safe_str(detailed_state).lower()

```
confirmed_status_codes = {
    "F",    # Final
    "O",    # Game Over / Over
    "I",    # In Progress
    "U",    # Suspended / Unknown game state after start
    "W",    # Warmup / delayed states can appear after lineups
    "DR",   # Delayed Rain
    "DI",   # Delayed
    "C",    # Completed
    "CD",   # Completed Early / Completed Delayed
}

pregame_keywords = (
    "preview",
    "scheduled",
    "pre-game",
    "pregame",
    "warmup",
)

live_or_final_keywords = (
    "in progress",
    "final",
    "game over",
    "delayed",
    "suspended",
    "completed",
)

is_pregame = any(keyword in detailed_lower for keyword in pregame_keywords)
is_live_or_final = (
    status_upper in confirmed_status_codes
    or any(keyword in detailed_lower for keyword in live_or_final_keywords)
)

return bool(is_live_or_final and not is_pregame)
```

def _extract_first_pitcher(
pitcher_entries: Any,
players_map: Any,
game_status: str,
detailed_state: str,
) -> Tuple[Optional[int], str, bool]:
"""Extract first pitcher from MLB boxscore pitcher list.

```
Returns:
- pitcher_id
- pitcher_name
- confirmed starter flag

The confirmed flag is intentionally conservative. Pregame probable pitchers
should not be treated as confirmed starters.
"""
pitcher_ids = _extract_player_ids_from_entries(pitcher_entries)
if not pitcher_ids:
    return None, "", False

pitcher_id = pitcher_ids[0]
player_obj = _get_player_from_players_map(players_map, pitcher_id)
pitcher_name = _extract_player_name(player_obj)

confirmed = (
    pitcher_id is not None
    and _game_has_started_or_finished(game_status, detailed_state)
)

return pitcher_id, pitcher_name, confirmed
```

def _find_catcher_from_order(
batting_ids: List[int],
players_map: Any,
) -> Tuple[Optional[int], str]:
"""Find catcher from batting order using players map."""
for player_id in batting_ids:
player_obj = _get_player_from_players_map(players_map, player_id)
if not player_obj:
continue

```
    position = player_obj.get("position") or {}
    code = _safe_str(position.get("code"))
    abbreviation = _safe_str(position.get("abbreviation")).upper()

    if code == "2" or abbreviation == "C":
        return player_id, _extract_player_name(player_obj)

return None, ""
```

def _extract_home_plate_umpire(boxscore: Any) -> Tuple[Optional[int], str]:
"""Extract home plate umpire from MLB boxscore officials."""
officials = _safe_get(boxscore, "officials", default=[])

```
if not isinstance(officials, list):
    return None, ""

for official in officials:
    if not isinstance(official, dict):
        continue

    official_type = _safe_str(official.get("officialType")).lower()
    assignment = _safe_str(official.get("assignment")).lower()

    is_home_plate = (
        "home plate" in official_type
        or official_type == "hp"
        or "home plate" in assignment
        or assignment == "hp"
    )

    if not is_home_plate:
        continue

    umpire_id = (
        _safe_int(_safe_get(official, "official.id"))
        or _safe_int(official.get("id"))
    )
    umpire_name = (
        _safe_str(_safe_get(official, "official.fullName"))
        or _safe_str(official.get("fullName"))
    )

    return umpire_id, umpire_name

return None, ""
```

# ---------------------------------------------------------------------------

# Row builders

# ---------------------------------------------------------------------------

def _game_feed_fallback_row(game_id: Union[int, str], error_msg: str) -> Dict[str, Any]:
"""Create a complete fallback row when game feed cannot be fetched."""
captured_at = datetime.now(timezone.utc).isoformat()

```
return {
    "game_id": str(game_id),
    "game_feed_available": False,
    "game_feed_error": _safe_str(error_msg),
    "game_date": "",
    "game_status": "",
    "detailed_state": "",
    "start_time": "",
    "home_team_id": None,
    "away_team_id": None,
    "home_team_name": "",
    "away_team_name": "",
    "venue_id": None,
    "venue_name": "",
    "weather_temp": None,
    "weather_condition": "",
    "wind_speed": None,
    "wind_direction": "",
    "home_probable_pitcher_id": None,
    "away_probable_pitcher_id": None,
    "home_probable_pitcher_name": "",
    "away_probable_pitcher_name": "",
    "home_starting_pitcher_id": None,
    "away_starting_pitcher_id": None,
    "home_starting_pitcher_name": "",
    "away_starting_pitcher_name": "",
    "home_starting_pitcher_confirmed": False,
    "away_starting_pitcher_confirmed": False,
    "home_lineup_confirmed": False,
    "away_lineup_confirmed": False,
    "home_lineup_player_count": 0,
    "away_lineup_player_count": 0,
    "home_batting_order_ids": "",
    "away_batting_order_ids": "",
    "home_top3_player_ids": "",
    "away_top3_player_ids": "",
    "home_catcher_id": None,
    "away_catcher_id": None,
    "home_catcher_name": "",
    "away_catcher_name": "",
    "umpire_home_plate_id": None,
    "umpire_home_plate_name": "",
    "game_feed_captured_at": captured_at,
}
```

def _parse_feed_data(game_id: Union[int, str], payload: Dict[str, Any]) -> Dict[str, Any]:
"""Parse MLB feed/live or wrapped boxscore payload into a standard row."""
captured_at = datetime.now(timezone.utc).isoformat()

```
game_data = _safe_get(payload, "gameData", default={})
live_data = _safe_get(payload, "liveData", default={})
boxscore = _safe_get(live_data, "boxscore", default={})
box_teams = _safe_get(boxscore, "teams", default={})

# -----------------------------------------------------------------------
# Game metadata
# -----------------------------------------------------------------------
datetime_data = _safe_get(game_data, "datetime", default={})
status_data = _safe_get(game_data, "status", default={})
teams_data = _safe_get(game_data, "teams", default={})
venue_data = _safe_get(game_data, "venue", default={})
weather_data = _safe_get(game_data, "weather", default={})

game_date = _safe_str(_safe_get(datetime_data, "officialDate"))
start_time = _safe_str(_safe_get(datetime_data, "dateTime"))
game_status = _safe_str(_safe_get(status_data, "statusCode"))
detailed_state = _safe_str(_safe_get(status_data, "detailedState"))

home_team = _safe_get(teams_data, "home", default={})
away_team = _safe_get(teams_data, "away", default={})

home_team_id = _safe_int(_safe_get(home_team, "id"))
away_team_id = _safe_int(_safe_get(away_team, "id"))
home_team_name = _safe_str(_safe_get(home_team, "name"))
away_team_name = _safe_str(_safe_get(away_team, "name"))

if not isinstance(venue_data, dict) or not venue_data:
    venue_data = _safe_get(home_team, "venue", default={})

venue_id = _safe_int(_safe_get(venue_data, "id"))
venue_name = _safe_str(_safe_get(venue_data, "name"))

weather_temp = _safe_float(_safe_get(weather_data, "temp"))
weather_condition = _safe_str(_safe_get(weather_data, "condition"))
wind_text = _safe_str(_safe_get(weather_data, "wind"))
wind_speed, wind_direction = _parse_wind(wind_text)

# -----------------------------------------------------------------------
# Probable pitchers
# -----------------------------------------------------------------------
probable_pitchers = _safe_get(game_data, "probablePitchers", default={})
home_probable = _safe_get(probable_pitchers, "home", default={})
away_probable = _safe_get(probable_pitchers, "away", default={})

home_probable_pitcher_id = _safe_int(_safe_get(home_probable, "id"))
away_probable_pitcher_id = _safe_int(_safe_get(away_probable, "id"))
home_probable_pitcher_name = _safe_str(_safe_get(home_probable, "fullName"))
away_probable_pitcher_name = _safe_str(_safe_get(away_probable, "fullName"))

# -----------------------------------------------------------------------
# Boxscore teams and players maps
# -----------------------------------------------------------------------
home_box = _safe_get(box_teams, "home", default={})
away_box = _safe_get(box_teams, "away", default={})

home_players_map = _safe_get(home_box, "players", default={})
away_players_map = _safe_get(away_box, "players", default={})

# -----------------------------------------------------------------------
# Starting pitchers
# -----------------------------------------------------------------------
home_starting_pitcher_id, home_starting_pitcher_name, home_starting_pitcher_confirmed = (
    _extract_first_pitcher(
        _safe_get(home_box, "pitchers", default=[]),
        home_players_map,
        game_status,
        detailed_state,
    )
)

away_starting_pitcher_id, away_starting_pitcher_name, away_starting_pitcher_confirmed = (
    _extract_first_pitcher(
        _safe_get(away_box, "pitchers", default=[]),
        away_players_map,
        game_status,
        detailed_state,
    )
)

# -----------------------------------------------------------------------
# Batting order / lineup
# -----------------------------------------------------------------------
home_batting_order_raw = _safe_get(home_box, "battingOrder", default=[])
away_batting_order_raw = _safe_get(away_box, "battingOrder", default=[])
home_batters_raw = _safe_get(home_box, "batters", default=[])
away_batters_raw = _safe_get(away_box, "batters", default=[])

if not isinstance(home_batting_order_raw, list):
    home_batting_order_raw = []
if not isinstance(away_batting_order_raw, list):
    away_batting_order_raw = []
if not isinstance(home_batters_raw, list):
    home_batters_raw = []
if not isinstance(away_batters_raw, list):
    away_batters_raw = []

home_batting_order_ids = _extract_player_ids_from_entries(home_batting_order_raw)
away_batting_order_ids = _extract_player_ids_from_entries(away_batting_order_raw)

home_lineup_from_batters = False
away_lineup_from_batters = False

if not home_batting_order_ids and home_batters_raw:
    home_batting_order_ids = _extract_player_ids_from_entries(home_batters_raw)
    home_lineup_from_batters = True

if not away_batting_order_ids and away_batters_raw:
    away_batting_order_ids = _extract_player_ids_from_entries(away_batters_raw)
    away_lineup_from_batters = True

home_lineup_confirmed = (
    not home_lineup_from_batters
    and len(home_batting_order_ids) >= 9
)
away_lineup_confirmed = (
    not away_lineup_from_batters
    and len(away_batting_order_ids) >= 9
)

home_lineup_player_count = len(home_batting_order_ids)
away_lineup_player_count = len(away_batting_order_ids)

home_batting_order_ids_str = ",".join(str(player_id) for player_id in home_batting_order_ids)
away_batting_order_ids_str = ",".join(str(player_id) for player_id in away_batting_order_ids)

home_top3_player_ids = home_batting_order_ids[:3]
away_top3_player_ids = away_batting_order_ids[:3]

home_top3_player_ids_str = ",".join(str(player_id) for player_id in home_top3_player_ids)
away_top3_player_ids_str = ",".join(str(player_id) for player_id in away_top3_player_ids)

home_catcher_id, home_catcher_name = _find_catcher_from_order(
    home_batting_order_ids,
    home_players_map,
)
away_catcher_id, away_catcher_name = _find_catcher_from_order(
    away_batting_order_ids,
    away_players_map,
)

# -----------------------------------------------------------------------
# Officials
# -----------------------------------------------------------------------
umpire_home_plate_id, umpire_home_plate_name = _extract_home_plate_umpire(boxscore)

return {
    "game_id": str(game_id),
    "game_feed_available": True,
    "game_feed_error": "",
    "game_date": game_date,
    "game_status": game_status,
    "detailed_state": detailed_state,
    "start_time": start_time,
    "home_team_id": home_team_id,
    "away_team_id": away_team_id,
    "home_team_name": home_team_name,
    "away_team_name": away_team_name,
    "venue_id": venue_id,
    "venue_name": venue_name,
    "weather_temp": weather_temp,
    "weather_condition": weather_condition,
    "wind_speed": wind_speed,
    "wind_direction": wind_direction,
    "home_probable_pitcher_id": home_probable_pitcher_id,
    "away_probable_pitcher_id": away_probable_pitcher_id,
    "home_probable_pitcher_name": home_probable_pitcher_name,
    "away_probable_pitcher_name": away_probable_pitcher_name,
    "home_starting_pitcher_id": home_starting_pitcher_id,
    "away_starting_pitcher_id": away_starting_pitcher_id,
    "home_starting_pitcher_name": home_starting_pitcher_name,
    "away_starting_pitcher_name": away_starting_pitcher_name,
    "home_starting_pitcher_confirmed": home_starting_pitcher_confirmed,
    "away_starting_pitcher_confirmed": away_starting_pitcher_confirmed,
    "home_lineup_confirmed": home_lineup_confirmed,
    "away_lineup_confirmed": away_lineup_confirmed,
    "home_lineup_player_count": home_lineup_player_count,
    "away_lineup_player_count": away_lineup_player_count,
    "home_batting_order_ids": home_batting_order_ids_str,
    "away_batting_order_ids": away_batting_order_ids_str,
    "home_top3_player_ids": home_top3_player_ids_str,
    "away_top3_player_ids": away_top3_player_ids_str,
    "home_catcher_id": home_catcher_id,
    "away_catcher_id": away_catcher_id,
    "home_catcher_name": home_catcher_name,
    "away_catcher_name": away_catcher_name,
    "umpire_home_plate_id": umpire_home_plate_id,
    "umpire_home_plate_name": umpire_home_plate_name,
    "game_feed_captured_at": captured_at,
}
```

# ---------------------------------------------------------------------------

# Public API

# ---------------------------------------------------------------------------

def fetch_mlb_game_feed_context(
game_id: Union[int, str],
errors: Optional[List[str]] = None,
timeout: int = 15,
) -> Dict[str, Any]:
"""Fetch one game context row from MLB Stats API.

```
The live feed endpoint is attempted first. If it fails, the boxscore endpoint
is attempted as a partial fallback.
"""
game_id_text = str(game_id)

def append_error(message: str) -> None:
    if errors is not None:
        errors.append(f"MLB game feed error for {game_id_text}: {message}")

# First attempt: full live feed.
live_url = MLB_GAME_FEED_URL.format(game_id=game_id_text)

try:
    response = requests.get(live_url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError("Live feed response was not a JSON object")

    return _parse_feed_data(game_id_text, payload)

except Exception as live_error:
    live_error_text = str(live_error)

# Second attempt: partial boxscore fallback.
boxscore_url = MLB_BOXSCORE_URL.format(game_id=game_id_text)

try:
    response = requests.get(boxscore_url, timeout=timeout)
    response.raise_for_status()
    boxscore_payload = response.json()

    if not isinstance(boxscore_payload, dict):
        raise ValueError("Boxscore response was not a JSON object")

    home_team_data = _safe_get(boxscore_payload, "teams.home.team", default={})
    away_team_data = _safe_get(boxscore_payload, "teams.away.team", default={})

    wrapped_payload = {
        "gameData": {
            "teams": {
                "home": home_team_data,
                "away": away_team_data,
            },
            "venue": _safe_get(home_team_data, "venue", default={}),
            "datetime": {
                "officialDate": "",
                "dateTime": "",
            },
            "status": {
                "statusCode": "",
                "detailedState": "",
            },
            "probablePitchers": {},
            "weather": {},
        },
        "liveData": {
            "boxscore": boxscore_payload,
        },
    }

    row = _parse_feed_data(game_id_text, wrapped_payload)
    row["game_feed_error"] = f"live feed failed; boxscore fallback used: {live_error_text}"
    return row

except Exception as boxscore_error:
    error_message = (
        f"live feed failed: {live_error_text}; "
        f"boxscore fallback failed: {boxscore_error}"
    )
    append_error(error_message)
    return _game_feed_fallback_row(game_id_text, error_message)
```

def fetch_mlb_game_feed_contexts(
game_ids: List[Union[int, str]],
errors: Optional[List[str]] = None,
timeout: int = 15,
sleep_seconds: float = 0.1,
) -> pd.DataFrame:
"""Fetch multiple game context rows and return them as a DataFrame."""
rows: List[Dict[str, Any]] = []

```
for game_id in game_ids:
    rows.append(
        fetch_mlb_game_feed_context(
            game_id,
            errors=errors,
            timeout=timeout,
        )
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

return pd.DataFrame(rows)
```

if **name** == "**main**":
sample_game_ids = [717426]
sample_errors: List[str] = []
sample_df = fetch_mlb_game_feed_contexts(
sample_game_ids,
errors=sample_errors,
timeout=20,
)
print("Errors:", sample_errors)
print(sample_df.head())
