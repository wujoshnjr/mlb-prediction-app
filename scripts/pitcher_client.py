from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PERSON_URL = "https://statsapi.mlb.com/api/v1/people/{person_id}"
MLB_PERSON_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{person_id}/stats"

REQUEST_TIMEOUT = 20
PITCHER_SOURCE = "mlb_statsapi_probable_pitcher"

PITCHER_COLUMNS = [
    "game_id",
    "game_date",
    "start_time",
    "game_status",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "home_pitcher_id",
    "away_pitcher_id",
    "home_probable_pitcher_id",
    "away_probable_pitcher_id",
    "home_probable_pitcher_name",
    "away_probable_pitcher_name",
    "home_probable_pitcher_available",
    "away_probable_pitcher_available",
    "home_starting_pitcher_confirmed",
    "away_starting_pitcher_confirmed",
    "home_era",
    "away_era",
    "home_fip",
    "away_fip",
    "home_whip",
    "away_whip",
    "home_k_per_9",
    "away_k_per_9",
    "home_bb_per_9",
    "away_bb_per_9",
    "home_pitch_hand",
    "away_pitch_hand",
    "home_stuff_plus",
    "away_stuff_plus",
    "home_csw_pct",
    "away_csw_pct",
    "pitcher_source",
    "pitcher_fetched_at",
    "pitcher_metrics_quality",
]


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format ending in Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _log_error(errors: Optional[List[str]], message: str) -> None:
    """Append an error message when an error list was supplied."""
    if errors is not None:
        errors.append(message)


def _fetch_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
    label: str = "request",
) -> Optional[Dict[str, Any]]:
    """Fetch one JSON object without raising an exception to the caller."""
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


def _safe_float(value: Any) -> Optional[float]:
    """Convert a numeric value to float, preserving missing values as None."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Convert a value to int, returning None when unavailable."""
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_probable_pitcher(
    team_payload: Dict[str, Any],
) -> tuple[Optional[int], str]:
    """Extract one probable pitcher id and name from schedule team data."""
    if not isinstance(team_payload, dict):
        return None, ""

    probable = team_payload.get("probablePitcher", {})
    if not isinstance(probable, dict):
        return None, ""

    pitcher_id = _safe_int(probable.get("id"))
    pitcher_name = str(probable.get("fullName") or "")

    return pitcher_id, pitcher_name


def _fetch_pitcher_profile(
    pitcher_id: Optional[int],
    *,
    season: int,
    errors: Optional[List[str]],
) -> Dict[str, Any]:
    """Fetch official basic pitcher profile and season statistics.

    This function does not invent Stuff+ or CSW values. Those fields remain
    empty until an audited source is added.
    """
    empty_profile: Dict[str, Any] = {
        "name": "",
        "pitch_hand": None,
        "era": None,
        "fip": None,
        "whip": None,
        "k_per_9": None,
        "bb_per_9": None,
        "stuff_plus": None,
        "csw_pct": None,
    }

    if pitcher_id is None:
        return empty_profile

    profile = dict(empty_profile)

    person_payload = _fetch_json(
        MLB_PERSON_URL.format(person_id=pitcher_id),
        errors=errors,
        label=f"MLB pitcher profile fetch for {pitcher_id}",
    )

    if person_payload is not None:
        people = person_payload.get("people", [])
        if isinstance(people, list) and people:
            person = people[0] if isinstance(people[0], dict) else {}
            profile["name"] = str(person.get("fullName") or "")
            pitch_hand = person.get("pitchHand", {})
            if isinstance(pitch_hand, dict):
                profile["pitch_hand"] = (
                    str(pitch_hand.get("code"))
                    if pitch_hand.get("code")
                    else None
                )

    stats_payload = _fetch_json(
        MLB_PERSON_STATS_URL.format(person_id=pitcher_id),
        params={
            "stats": "season",
            "group": "pitching",
            "season": season,
            "gameType": "R",
        },
        errors=errors,
        label=f"MLB pitcher season stats fetch for {pitcher_id}",
    )

    if stats_payload is None:
        return profile

    stats_groups = stats_payload.get("stats", [])
    if not isinstance(stats_groups, list):
        return profile

    season_stat: Dict[str, Any] = {}

    for stats_group in stats_groups:
        if not isinstance(stats_group, dict):
            continue
        splits = stats_group.get("splits", [])
        if not isinstance(splits, list) or not splits:
            continue
        first_split = splits[0] if isinstance(splits[0], dict) else {}
        stat_payload = first_split.get("stat", {})
        if isinstance(stat_payload, dict):
            season_stat = stat_payload
            break

    if not season_stat:
        return profile

    profile["era"] = _safe_float(season_stat.get("era"))
    profile["fip"] = _safe_float(season_stat.get("fip"))
    profile["whip"] = _safe_float(season_stat.get("whip"))
    profile["k_per_9"] = _safe_float(
        season_stat.get("strikeOutsPer9Inn")
    )
    profile["bb_per_9"] = _safe_float(
        season_stat.get("walksPer9Inn")
    )

    return profile


def fetch_probable_pitchers(
    date_str: Optional[str] = None,
    errors: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fetch official probable pitcher data for all scheduled MLB games.

    Every scheduled game returns one row, even if one or both probable
    pitchers are not available yet. Probable pitchers are not treated as
    confirmed starting pitchers.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        season = int(str(date_str)[:4])
    except (TypeError, ValueError):
        season = datetime.now(timezone.utc).year

    fetched_at = _utc_now_iso()

    schedule_payload = _fetch_json(
        MLB_SCHEDULE_URL,
        params={
            "sportId": 1,
            "date": date_str,
            "hydrate": "probablePitcher",
            "gameTypes": "R",
        },
        errors=errors,
        label="MLB probable pitcher schedule fetch",
    )

    if schedule_payload is None:
        return pd.DataFrame(columns=PITCHER_COLUMNS)

    rows: List[Dict[str, Any]] = []

    for date_payload in schedule_payload.get("dates", []) or []:
        if not isinstance(date_payload, dict):
            continue

        for game in date_payload.get("games", []) or []:
            if not isinstance(game, dict):
                continue

            game_id = game.get("gamePk")
            if game_id is None:
                _log_error(
                    errors,
                    "MLB probable pitcher schedule row missing gamePk.",
                )
                continue

            teams = game.get("teams", {})
            if not isinstance(teams, dict):
                teams = {}

            home_payload = teams.get("home", {})
            away_payload = teams.get("away", {})
            if not isinstance(home_payload, dict):
                home_payload = {}
            if not isinstance(away_payload, dict):
                away_payload = {}

            home_team_payload = home_payload.get("team", {})
            away_team_payload = away_payload.get("team", {})
            if not isinstance(home_team_payload, dict):
                home_team_payload = {}
            if not isinstance(away_team_payload, dict):
                away_team_payload = {}

            home_pitcher_id, home_pitcher_name = _extract_probable_pitcher(
                home_payload
            )
            away_pitcher_id, away_pitcher_name = _extract_probable_pitcher(
                away_payload
            )

            home_profile = _fetch_pitcher_profile(
                home_pitcher_id,
                season=season,
                errors=errors,
            )
            away_profile = _fetch_pitcher_profile(
                away_pitcher_id,
                season=season,
                errors=errors,
            )

            if not home_pitcher_name:
                home_pitcher_name = str(home_profile.get("name") or "")
            if not away_pitcher_name:
                away_pitcher_name = str(away_profile.get("name") or "")

            status_payload = game.get("status", {})
            if not isinstance(status_payload, dict):
                status_payload = {}

            rows.append(
                {
                    "game_id": game_id,
                    "game_date": date_str,
                    "start_time": str(game.get("gameDate") or ""),
                    "game_status": str(
                        status_payload.get("abstractGameState")
                        or status_payload.get("detailedState")
                        or ""
                    ),
                    "home_team_id": home_team_payload.get("id"),
                    "away_team_id": away_team_payload.get("id"),
                    "home_team": str(home_team_payload.get("name") or ""),
                    "away_team": str(away_team_payload.get("name") or ""),
                    "home_pitcher_id": home_pitcher_id,
                    "away_pitcher_id": away_pitcher_id,
                    "home_probable_pitcher_id": home_pitcher_id,
                    "away_probable_pitcher_id": away_pitcher_id,
                    "home_probable_pitcher_name": home_pitcher_name,
                    "away_probable_pitcher_name": away_pitcher_name,
                    "home_probable_pitcher_available": (
                        home_pitcher_id is not None
                    ),
                    "away_probable_pitcher_available": (
                        away_pitcher_id is not None
                    ),
                    "home_starting_pitcher_confirmed": None,
                    "away_starting_pitcher_confirmed": None,
                    "home_era": home_profile.get("era"),
                    "away_era": away_profile.get("era"),
                    "home_fip": home_profile.get("fip"),
                    "away_fip": away_profile.get("fip"),
                    "home_whip": home_profile.get("whip"),
                    "away_whip": away_profile.get("whip"),
                    "home_k_per_9": home_profile.get("k_per_9"),
                    "away_k_per_9": away_profile.get("k_per_9"),
                    "home_bb_per_9": home_profile.get("bb_per_9"),
                    "away_bb_per_9": away_profile.get("bb_per_9"),
                    "home_pitch_hand": home_profile.get("pitch_hand"),
                    "away_pitch_hand": away_profile.get("pitch_hand"),
                    "home_stuff_plus": None,
                    "away_stuff_plus": None,
                    "home_csw_pct": None,
                    "away_csw_pct": None,
                    "pitcher_source": PITCHER_SOURCE,
                    "pitcher_fetched_at": fetched_at,
                    "pitcher_metrics_quality": (
                        "official_basic_stats_only_no_stuff_plus_or_csw"
                    ),
                }
            )

    return pd.DataFrame(rows, columns=PITCHER_COLUMNS)
