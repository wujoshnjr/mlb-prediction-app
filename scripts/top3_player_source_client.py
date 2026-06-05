from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

MLB_BASE_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 15

OUTPUT_COLUMNS = [
    "game_id",
    "game_date",
    "start_time",
    "captured_at",
    "home_team",
    "away_team",
    "home_top3_player_ids",
    "away_top3_player_ids",
    "home_top3_player_names",
    "away_top3_player_names",
    "home_top3_source_status",
    "away_top3_source_status",
    "home_top3_reason",
    "away_top3_reason",
    "top3_player_source",
    "top3_player_captured_at",
]

TEAM_ID_MAP: Dict[str, int] = {
    "Arizona Diamondbacks": 109,
    "Athletics": 133,
    "Oakland Athletics": 133,
    "Atlanta Braves": 144,
    "Baltimore Orioles": 110,
    "Boston Red Sox": 111,
    "Chicago Cubs": 112,
    "Chicago White Sox": 145,
    "Cincinnati Reds": 113,
    "Cleveland Guardians": 114,
    "Colorado Rockies": 115,
    "Detroit Tigers": 116,
    "Houston Astros": 117,
    "Kansas City Royals": 118,
    "Los Angeles Angels": 108,
    "Los Angeles Dodgers": 119,
    "Miami Marlins": 146,
    "Milwaukee Brewers": 158,
    "Minnesota Twins": 142,
    "New York Mets": 121,
    "New York Yankees": 147,
    "Philadelphia Phillies": 143,
    "Pittsburgh Pirates": 134,
    "San Diego Padres": 135,
    "San Francisco Giants": 137,
    "Seattle Mariners": 136,
    "St. Louis Cardinals": 138,
    "Tampa Bay Rays": 139,
    "Texas Rangers": 140,
    "Toronto Blue Jays": 141,
    "Washington Nationals": 120,
}

TEAM_ALIASES: Dict[str, str] = {
    "ari": "Arizona Diamondbacks",
    "diamondbacks": "Arizona Diamondbacks",
    "dbacks": "Arizona Diamondbacks",
    "d-backs": "Arizona Diamondbacks",
    "ath": "Athletics",
    "oak": "Oakland Athletics",
    "athletics": "Athletics",
    "oakland athletics": "Oakland Athletics",
    "a's": "Athletics",
    "as": "Athletics",
    "atl": "Atlanta Braves",
    "braves": "Atlanta Braves",
    "bal": "Baltimore Orioles",
    "orioles": "Baltimore Orioles",
    "bos": "Boston Red Sox",
    "red sox": "Boston Red Sox",
    "boston red sox": "Boston Red Sox",
    "chc": "Chicago Cubs",
    "cubs": "Chicago Cubs",
    "cws": "Chicago White Sox",
    "white sox": "Chicago White Sox",
    "cin": "Cincinnati Reds",
    "reds": "Cincinnati Reds",
    "cle": "Cleveland Guardians",
    "guardians": "Cleveland Guardians",
    "col": "Colorado Rockies",
    "rockies": "Colorado Rockies",
    "det": "Detroit Tigers",
    "tigers": "Detroit Tigers",
    "hou": "Houston Astros",
    "astros": "Houston Astros",
    "kc": "Kansas City Royals",
    "kcr": "Kansas City Royals",
    "royals": "Kansas City Royals",
    "laa": "Los Angeles Angels",
    "angels": "Los Angeles Angels",
    "lad": "Los Angeles Dodgers",
    "dodgers": "Los Angeles Dodgers",
    "mia": "Miami Marlins",
    "marlins": "Miami Marlins",
    "mil": "Milwaukee Brewers",
    "brewers": "Milwaukee Brewers",
    "min": "Minnesota Twins",
    "twins": "Minnesota Twins",
    "nym": "New York Mets",
    "mets": "New York Mets",
    "nyy": "New York Yankees",
    "yankees": "New York Yankees",
    "phi": "Philadelphia Phillies",
    "phillies": "Philadelphia Phillies",
    "pit": "Pittsburgh Pirates",
    "pirates": "Pittsburgh Pirates",
    "sd": "San Diego Padres",
    "sdp": "San Diego Padres",
    "padres": "San Diego Padres",
    "sf": "San Francisco Giants",
    "sfg": "San Francisco Giants",
    "giants": "San Francisco Giants",
    "sea": "Seattle Mariners",
    "mariners": "Seattle Mariners",
    "stl": "St. Louis Cardinals",
    "cardinals": "St. Louis Cardinals",
    "st louis cardinals": "St. Louis Cardinals",
    "saint louis cardinals": "St. Louis Cardinals",
    "tb": "Tampa Bay Rays",
    "tbr": "Tampa Bay Rays",
    "rays": "Tampa Bay Rays",
    "tex": "Texas Rangers",
    "rangers": "Texas Rangers",
    "tor": "Toronto Blue Jays",
    "blue jays": "Toronto Blue Jays",
    "wsh": "Washington Nationals",
    "was": "Washington Nationals",
    "nationals": "Washington Nationals",
}


def _current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _safe_float(value: Any) -> Optional[float]:
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
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _normalise_key(value: Any) -> str:
    text = _safe_str(value).lower()
    text = text.replace("&", "and").replace(".", "").replace("-", " ")
    text = text.replace("_", " ").replace("'", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalise_team_name(value: Any) -> str:
    raw = _safe_str(value)
    if not raw:
        return ""

    key = _normalise_key(raw)
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]

    for full_name in TEAM_ID_MAP:
        if _normalise_key(full_name) == key:
            return full_name

    for alias, full_name in TEAM_ALIASES.items():
        if key == alias or key in alias or alias in key:
            return full_name

    return raw


def team_id_from_name(value: Any) -> Optional[int]:
    team_name = normalise_team_name(value)
    return TEAM_ID_MAP.get(team_name)


def _season_from_game_date(value: Any) -> int:
    text = _safe_str(value)
    if text:
        try:
            return int(text[:4])
        except Exception:
            pass
    return datetime.now(timezone.utc).year


def parse_player_ids(value: Any) -> List[int]:
    if value is None:
        return []

    if isinstance(value, list):
        result = []
        for item in value:
            item_int = _safe_int(item)
            if item_int is not None:
                result.append(item_int)
        return result

    text = _safe_str(value)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parse_player_ids(parsed)
    except Exception:
        pass

    ids: List[int] = []
    for token in re.split(r"[,\s;|]+", text):
        token = token.strip()
        if token.isdigit():
            ids.append(int(token))
    return ids


def _latest_context_rows(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["game_id"] = working["game_id"].astype(str)

    if "captured_at" in working.columns:
        working["captured_at_dt"] = pd.to_datetime(
            working["captured_at"],
            errors="coerce",
            utc=True,
        )
        if working["captured_at_dt"].notna().any():
            return (
                working.sort_values(["game_id", "captured_at_dt"])
                .groupby("game_id", as_index=False)
                .tail(1)
            )

    return working.drop_duplicates(subset=["game_id"], keep="last")


def _empty_output(output_path: Optional[str]) -> pd.DataFrame:
    frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
    return frame


def _clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame.where(pd.notnull(frame), "")
    return frame


def _request_json(url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        return None, f"request_failed: {exc}"

    if response.status_code != 200:
        return None, f"http_{response.status_code}: {response.text[:160]}"

    try:
        data = response.json()
    except Exception as exc:
        return None, f"json_error: {exc}"

    if not isinstance(data, dict):
        return None, "json_not_object"

    return data, ""


def _fetch_roster(
    team_id: int,
    roster_cache: Dict[int, Tuple[Optional[List[Dict[str, Any]]], str]],
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    if team_id in roster_cache:
        return roster_cache[team_id]

    url = f"{MLB_BASE_URL}/teams/{team_id}/roster"
    data, error = _request_json(url, params={"rosterType": "active"})

    if data is None:
        result = (None, error)
    else:
        roster = data.get("roster")
        if isinstance(roster, list):
            result = (roster, "")
        else:
            result = (None, "roster_missing")

    roster_cache[team_id] = result
    return result


def _fetch_player_stats(
    player_ids: List[int],
    season: int,
    stats_cache: Dict[Tuple[int, Tuple[int, ...]], Tuple[Optional[Dict[int, Dict[str, Any]]], str]],
) -> Tuple[Optional[Dict[int, Dict[str, Any]]], str]:
    unique_ids = tuple(sorted(set(int(player_id) for player_id in player_ids)))
    cache_key = (int(season), unique_ids)

    if cache_key in stats_cache:
        return stats_cache[cache_key]

    if not unique_ids:
        result = ({}, "")
        stats_cache[cache_key] = result
        return result

    url = f"{MLB_BASE_URL}/people"
    params = {
        "personIds": ",".join(str(player_id) for player_id in unique_ids),
        "hydrate": f"stats(group=[hitting],type=season,season={season})",
    }

    data, error = _request_json(url, params=params)
    if data is None:
        result = (None, error)
        stats_cache[cache_key] = result
        return result

    people = data.get("people")
    if not isinstance(people, list):
        result = (None, "people_missing")
        stats_cache[cache_key] = result
        return result

    output: Dict[int, Dict[str, Any]] = {}
    for person in people:
        if not isinstance(person, dict):
            continue

        player_id = _safe_int(person.get("id"))
        if player_id is None:
            continue

        stats_blocks = person.get("stats")
        if not isinstance(stats_blocks, list):
            continue

        for stats_block in stats_blocks:
            if not isinstance(stats_block, dict):
                continue

            splits = stats_block.get("splits")
            if isinstance(splits, list) and splits:
                stat = splits[0].get("stat")
                if isinstance(stat, dict):
                    output[player_id] = stat
                    break

    result = (output, "")
    stats_cache[cache_key] = result
    return result


def _position_player_rows(roster: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in roster:
        if not isinstance(row, dict):
            continue

        position = row.get("position") if isinstance(row.get("position"), dict) else {}
        position_type = _safe_str(position.get("type")).lower()
        position_code = _safe_str(position.get("code")).lower()

        if position_type == "pitcher" or position_code == "1":
            continue

        person = row.get("person") if isinstance(row.get("person"), dict) else {}
        player_id = _safe_int(person.get("id"))
        if player_id is None:
            continue

        rows.append(row)

    return rows


def _ops_from_stats(stats: Dict[str, Any]) -> Tuple[float, bool]:
    ops = _safe_float(stats.get("ops"))
    if ops is not None:
        return ops, True

    obp = _safe_float(stats.get("obp"))
    slg = _safe_float(stats.get("slg"))
    if obp is not None or slg is not None:
        return (obp or 0.0) + (slg or 0.0), True

    avg = _safe_float(stats.get("avg"))
    if avg is not None:
        return avg, True

    return 0.0, False


def _build_side_top3(
    existing_ids_raw: Any,
    team_name_raw: Any,
    game_date: str,
    roster_cache: Dict[int, Tuple[Optional[List[Dict[str, Any]]], str]],
    stats_cache: Dict[Tuple[int, Tuple[int, ...]], Tuple[Optional[Dict[int, Dict[str, Any]]], str]],
) -> Dict[str, Any]:
    existing_ids = parse_player_ids(existing_ids_raw)
    if len(existing_ids) >= 3:
        ids = existing_ids[:3]
        return {
            "ids": ids,
            "names": [str(player_id) for player_id in ids],
            "status": "daily_context",
            "reason": "Using existing top3 player ids from daily context",
        }

    team_name = normalise_team_name(team_name_raw)
    team_id = team_id_from_name(team_name)

    if team_id is None:
        return {
            "ids": [],
            "names": [],
            "status": "unavailable",
            "reason": f"Unknown team: {team_name_raw}",
        }

    roster, roster_error = _fetch_roster(team_id, roster_cache)
    if roster is None:
        return {
            "ids": [],
            "names": [],
            "status": "api_error",
            "reason": f"Roster API failed: {roster_error}",
        }

    position_players = _position_player_rows(roster)
    if len(position_players) < 3:
        return {
            "ids": [],
            "names": [],
            "status": "unavailable",
            "reason": f"Only {len(position_players)} position players found on roster",
        }

    player_ids = []
    player_name_by_id: Dict[int, str] = {}

    for row in position_players:
        person = row.get("person") if isinstance(row.get("person"), dict) else {}
        player_id = _safe_int(person.get("id"))
        if player_id is None:
            continue
        player_ids.append(player_id)
        player_name_by_id[player_id] = _safe_str(person.get("fullName"))

    season = _season_from_game_date(game_date)
    stats_map, stats_error = _fetch_player_stats(player_ids, season, stats_cache)

    if stats_map is None:
        return {
            "ids": [],
            "names": [],
            "status": "api_error",
            "reason": f"Hitting stats API failed: {stats_error}",
        }

    ranked: List[Tuple[int, float, int, bool, str]] = []
    for player_id in player_ids:
        stats = stats_map.get(player_id, {})
        if not isinstance(stats, dict):
            stats = {}

        ops_score, has_stats = _ops_from_stats(stats)
        plate_appearances = _safe_int(stats.get("plateAppearances")) or 0
        games_played = _safe_int(stats.get("gamesPlayed")) or 0

        ranked.append(
            (
                player_id,
                float(ops_score),
                int(plate_appearances or games_played),
                bool(has_stats),
                player_name_by_id.get(player_id, str(player_id)),
            )
        )

    ranked.sort(key=lambda item: (item[3], item[1], item[2]), reverse=True)
    top3 = ranked[:3]

    ids = [item[0] for item in top3]
    names = [item[4] for item in top3]
    players_with_stats = sum(1 for item in top3 if item[3])

    if len(ids) < 3:
        return {
            "ids": ids,
            "names": names,
            "status": "insufficient_data",
            "reason": f"Only {len(ids)} players ranked",
        }

    if players_with_stats == 0:
        return {
            "ids": ids,
            "names": names,
            "status": "api_roster_fallback",
            "reason": f"No season hitting stats found for {season}; using roster order fallback",
        }

    return {
        "ids": ids,
        "names": names,
        "status": "api_ops",
        "reason": f"Top3 selected by season hitting OPS/OBP+SLG for {season}; players_with_stats={players_with_stats}",
    }


def build_top3_player_context(
    daily_context_path: str = "data/daily_game_context.csv",
    output_path: Optional[str] = "data/top3_player_context.csv",
) -> pd.DataFrame:
    captured_at = _current_utc_iso()
    context_path = Path(daily_context_path)

    if not context_path.exists():
        return _empty_output(output_path)

    try:
        context_frame = pd.read_csv(context_path)
    except Exception:
        return _empty_output(output_path)

    if context_frame.empty:
        return _empty_output(output_path)

    for column in (
        "game_id",
        "game_date",
        "start_time",
        "captured_at",
        "home_team",
        "away_team",
        "home_top3_player_ids",
        "away_top3_player_ids",
    ):
        if column not in context_frame.columns:
            context_frame[column] = ""

    context_frame = context_frame.dropna(subset=["game_id"]).copy()
    if context_frame.empty:
        return _empty_output(output_path)

    latest_frame = _latest_context_rows(context_frame)

    roster_cache: Dict[int, Tuple[Optional[List[Dict[str, Any]]], str]] = {}
    stats_cache: Dict[Tuple[int, Tuple[int, ...]], Tuple[Optional[Dict[int, Dict[str, Any]]], str]] = {}

    rows: List[Dict[str, Any]] = []

    for _, row in latest_frame.iterrows():
        game_id = _safe_str(row.get("game_id"))
        game_date = _safe_str(row.get("game_date"))[:10]
        start_time = _safe_str(row.get("start_time"))
        home_team = _safe_str(row.get("home_team"))
        away_team = _safe_str(row.get("away_team"))

        home_result = _build_side_top3(
            row.get("home_top3_player_ids"),
            home_team,
            game_date,
            roster_cache,
            stats_cache,
        )
        away_result = _build_side_top3(
            row.get("away_top3_player_ids"),
            away_team,
            game_date,
            roster_cache,
            stats_cache,
        )

        rows.append(
            {
                "game_id": game_id,
                "game_date": game_date,
                "start_time": start_time,
                "captured_at": _safe_str(row.get("captured_at")),
                "home_team": home_team,
                "away_team": away_team,
                "home_top3_player_ids": ",".join(str(item) for item in home_result["ids"]),
                "away_top3_player_ids": ",".join(str(item) for item in away_result["ids"]),
                "home_top3_player_names": ",".join(str(item) for item in home_result["names"]),
                "away_top3_player_names": ",".join(str(item) for item in away_result["names"]),
                "home_top3_source_status": home_result["status"],
                "away_top3_source_status": away_result["status"],
                "home_top3_reason": home_result["reason"],
                "away_top3_reason": away_result["reason"],
                "top3_player_source": "daily_context_or_mlb_stats_api",
                "top3_player_captured_at": captured_at,
            }
        )

    output_frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_frame = _clean_dataframe(output_frame)

    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        output_frame.to_csv(destination, index=False)

    return output_frame


build_top3_context = build_top3_player_context


if __name__ == "__main__":
    df = build_top3_player_context()
    home_available = 0
    away_available = 0

    if not df.empty:
        home_available = int(
            df["home_top3_player_ids"].astype(str).str.split(",").map(
                lambda values: len([item for item in values if str(item).strip()])
            ).ge(3).sum()
        )
        away_available = int(
            df["away_top3_player_ids"].astype(str).str.split(",").map(
                lambda values: len([item for item in values if str(item).strip()])
            ).ge(3).sum()
        )

    print(
        json.dumps(
            {
                "rows": int(len(df)),
                "home_available": home_available,
                "away_available": away_available,
                "output_path": "data/top3_player_context.csv",
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        )
    )
