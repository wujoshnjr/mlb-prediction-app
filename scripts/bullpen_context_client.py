from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_LIVE_GAME_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"

REQUEST_TIMEOUT = 20
BULLPEN_SOURCE = "mlb_statsapi_completed_game_live_feed_v1"

BULLPEN_CONTEXT_COLUMNS = [
    "team_id",
    "team_name",
    "as_of_date",
    "captured_at",
    "bullpen_source",
    "bullpen_data_available",
    "completed_games_last_3d",
    "extra_innings_previous_game",
    "reliever_pitches_last_1d",
    "reliever_pitches_last_3d",
    "reliever_appearances_last_1d",
    "reliever_appearances_last_3d",
    "back_to_back_relievers_count",
    "three_in_three_days_relievers_count",
    "closer_candidate_pitcher_id",
    "closer_candidate_name",
    "closer_identification_method",
    "closer_used_last_1d",
    "closer_pitches_last_2d",
    "closer_available_estimate",
    "closer_available_estimate_reason",
    "bullpen_fatigue_score",
]


def _utc_now_iso() -> str:
    """Return the current UTC timestamp ending in Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _log_error(errors: Optional[List[str]], message: str) -> None:
    """Append a non-fatal error when an errors list was supplied."""
    if errors is not None:
        errors.append(message)


def _fetch_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
    label: str = "request",
) -> Optional[Dict[str, Any]]:
    """Fetch one JSON object without raising to the caller."""
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


def _parse_date(value: str) -> Optional[date]:
    """Parse a YYYY-MM-DD date string."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Convert a scalar to int while preserving unavailable values."""
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _fetch_schedule_games(
    game_date: str,
    *,
    errors: Optional[List[str]],
    label: str,
) -> tuple[List[Dict[str, Any]], bool]:
    """Fetch regular-season schedule games for one date."""
    payload = _fetch_json(
        MLB_SCHEDULE_URL,
        params={
            "sportId": 1,
            "date": game_date,
            "gameTypes": "R",
        },
        errors=errors,
        label=label,
    )

    if payload is None:
        return [], False

    games: List[Dict[str, Any]] = []

    for date_payload in payload.get("dates", []) or []:
        if not isinstance(date_payload, dict):
            continue

        for game in date_payload.get("games", []) or []:
            if isinstance(game, dict):
                games.append(game)

    return games, True


def _is_completed_game(game: Dict[str, Any]) -> bool:
    """Return True only for final or completed games."""
    status = game.get("status", {})
    if not isinstance(status, dict):
        return False

    abstract_state = str(status.get("abstractGameState") or "").lower()
    detailed_state = str(status.get("detailedState") or "").lower()

    return (
        abstract_state == "final"
        or detailed_state in {"final", "game over", "completed"}
    )


def _team_details_from_game(
    game: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Read home and away team identifiers from one schedule game."""
    teams = game.get("teams", {})
    if not isinstance(teams, dict):
        teams = {}

    output: Dict[str, Dict[str, Any]] = {}

    for side in ("home", "away"):
        side_payload = teams.get(side, {})
        if not isinstance(side_payload, dict):
            side_payload = {}

        team_payload = side_payload.get("team", {})
        if not isinstance(team_payload, dict):
            team_payload = {}

        output[side] = {
            "team_id": _safe_int(team_payload.get("id")),
            "team_name": str(team_payload.get("name") or ""),
        }

    return output


def _pitching_side_for_half_inning(half_inning: Any) -> Optional[str]:
    """Map batting half-inning to the defensive pitching team."""
    normalized = str(half_inning or "").strip().lower()

    if normalized == "top":
        return "home"

    if normalized == "bottom":
        return "away"

    return None


def _appearance_order_from_plays(
    live_payload: Dict[str, Any],
) -> Dict[str, List[int]]:
    """Recover actual pitcher appearance order from official play events."""
    appearance_order: Dict[str, List[int]] = {
        "home": [],
        "away": [],
    }
    seen: Dict[str, set[int]] = {
        "home": set(),
        "away": set(),
    }

    live_data = live_payload.get("liveData", {})
    if not isinstance(live_data, dict):
        return appearance_order

    plays_payload = live_data.get("plays", {})
    if not isinstance(plays_payload, dict):
        return appearance_order

    all_plays = plays_payload.get("allPlays", [])
    if not isinstance(all_plays, list):
        return appearance_order

    for play in all_plays:
        if not isinstance(play, dict):
            continue

        about = play.get("about", {})
        matchup = play.get("matchup", {})
        if not isinstance(about, dict) or not isinstance(matchup, dict):
            continue

        pitching_side = _pitching_side_for_half_inning(
            about.get("halfInning")
        )
        if pitching_side is None:
            continue

        pitcher_payload = matchup.get("pitcher", {})
        if not isinstance(pitcher_payload, dict):
            continue

        pitcher_id = _safe_int(pitcher_payload.get("id"))
        if pitcher_id is None or pitcher_id in seen[pitching_side]:
            continue

        seen[pitching_side].add(pitcher_id)
        appearance_order[pitching_side].append(pitcher_id)

    return appearance_order


def _player_pitch_count(
    players_payload: Dict[str, Any],
    pitcher_id: int,
) -> Optional[int]:
    """Read official pitches thrown by one pitcher from boxscore players."""
    player_payload = players_payload.get(f"ID{pitcher_id}", {})
    if not isinstance(player_payload, dict):
        return None

    stats_payload = player_payload.get("stats", {})
    if not isinstance(stats_payload, dict):
        return None

    pitching_payload = stats_payload.get("pitching", {})
    if not isinstance(pitching_payload, dict):
        return None

    return _safe_int(pitching_payload.get("numberOfPitches"))


def _actual_innings_from_live_feed(
    live_payload: Dict[str, Any],
) -> Optional[int]:
    """Read the actual inning count from a completed game payload."""
    live_data = live_payload.get("liveData", {})
    if not isinstance(live_data, dict):
        return None

    linescore = live_data.get("linescore", {})
    if not isinstance(linescore, dict):
        return None

    innings = linescore.get("innings", [])
    if not isinstance(innings, list) or not innings:
        return None

    return len(innings)


def _extract_game_bullpen_usage(
    live_payload: Dict[str, Any],
    *,
    game_id: Any,
    game_date: str,
    errors: Optional[List[str]],
) -> tuple[Dict[str, Optional[Dict[str, Any]]], Optional[int]]:
    """Extract reliable reliever usage for both teams in one completed game."""
    appearance_order = _appearance_order_from_plays(live_payload)

    live_data = live_payload.get("liveData", {})
    if not isinstance(live_data, dict):
        return {"home": None, "away": None}, None

    boxscore = live_data.get("boxscore", {})
    if not isinstance(boxscore, dict):
        return {"home": None, "away": None}, None

    teams_payload = boxscore.get("teams", {})
    if not isinstance(teams_payload, dict):
        return {"home": None, "away": None}, None

    extracted: Dict[str, Optional[Dict[str, Any]]] = {
        "home": None,
        "away": None,
    }

    for side in ("home", "away"):
        order = appearance_order.get(side, [])

        if not order:
            _log_error(
                errors,
                f"Game {game_id} {side}: no reliable pitcher appearance order.",
            )
            continue

        team_boxscore = teams_payload.get(side, {})
        if not isinstance(team_boxscore, dict):
            _log_error(
                errors,
                f"Game {game_id} {side}: missing boxscore team payload.",
            )
            continue

        players_payload = team_boxscore.get("players", {})
        if not isinstance(players_payload, dict):
            _log_error(
                errors,
                f"Game {game_id} {side}: missing boxscore player payload.",
            )
            continue

        reliever_ids = order[1:]
        relievers: List[Dict[str, Any]] = []
        complete = True

        for pitcher_id in reliever_ids:
            pitches_thrown = _player_pitch_count(
                players_payload,
                pitcher_id,
            )

            if pitches_thrown is None:
                complete = False
                _log_error(
                    errors,
                    (
                        f"Game {game_id} {side}: reliever {pitcher_id} "
                        "missing numberOfPitches."
                    ),
                )
                break

            relievers.append(
                {
                    "pitcher_id": pitcher_id,
                    "pitches_thrown": pitches_thrown,
                    "appearance_date": game_date,
                }
            )

        if not complete:
            continue

        extracted[side] = {
            "game_date": game_date,
            "relievers": relievers,
            "reliever_pitches": sum(
                row["pitches_thrown"] for row in relievers
            ),
            "reliever_appearances": len(relievers),
        }

    actual_innings = _actual_innings_from_live_feed(live_payload)

    return extracted, actual_innings


def fetch_bullpen_context(
    as_of_date: Optional[str] = None,
    errors: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build pregame bullpen fatigue context for teams playing on one date.

    Only completed regular-season games from the three calendar days before
    as_of_date are used. Data from as_of_date itself is never used.
    """
    if as_of_date is None:
        as_of_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prediction_date = _parse_date(as_of_date)
    if prediction_date is None:
        _log_error(
            errors,
            f"Invalid as_of_date for bullpen context: {as_of_date}",
        )
        return pd.DataFrame(columns=BULLPEN_CONTEXT_COLUMNS)

    captured_at = _utc_now_iso()

    target_games, target_schedule_ok = _fetch_schedule_games(
        as_of_date,
        errors=errors,
        label=f"MLB schedule fetch for bullpen target date {as_of_date}",
    )

    if not target_schedule_ok:
        return pd.DataFrame(columns=BULLPEN_CONTEXT_COLUMNS)

    target_teams: Dict[int, str] = {}

    for game in target_games:
        game_teams = _team_details_from_game(game)

        for side in ("home", "away"):
            team_id = game_teams[side]["team_id"]
            team_name = game_teams[side]["team_name"]

            if team_id is not None:
                target_teams[team_id] = team_name

    if not target_teams:
        return pd.DataFrame(columns=BULLPEN_CONTEXT_COLUMNS)

    lookback_dates = [
        (prediction_date - timedelta(days=days)).strftime("%Y-%m-%d")
        for days in (3, 2, 1)
    ]
    previous_day = lookback_dates[-1]

    accumulators: Dict[int, Dict[str, Any]] = {}

    for team_id, team_name in target_teams.items():
        accumulators[team_id] = {
            "team_name": team_name,
            "completed_games": 0,
            "has_data_issue": False,
            "pitches_by_date": defaultdict(int),
            "appearances_by_date": defaultdict(int),
            "reliever_dates": defaultdict(set),
            "previous_day_games": [],
        }

    entire_schedule_window_available = True

    for historical_date in lookback_dates:
        historical_games, schedule_ok = _fetch_schedule_games(
            historical_date,
            errors=errors,
            label=(
                "MLB schedule fetch for bullpen lookback date "
                f"{historical_date}"
            ),
        )

        if not schedule_ok:
            entire_schedule_window_available = False
            continue

        for game in historical_games:
            if not _is_completed_game(game):
                continue

            game_teams = _team_details_from_game(game)
            sides_for_targets: Dict[str, int] = {}

            for side in ("home", "away"):
                team_id = game_teams[side]["team_id"]
                if team_id in target_teams:
                    sides_for_targets[side] = team_id

            if not sides_for_targets:
                continue

            game_id = game.get("gamePk")
            if game_id is None:
                for team_id in sides_for_targets.values():
                    accumulators[team_id]["has_data_issue"] = True
                _log_error(
                    errors,
                    f"Completed game on {historical_date} missing gamePk.",
                )
                continue

            for team_id in sides_for_targets.values():
                accumulators[team_id]["completed_games"] += 1

            live_payload = _fetch_json(
                MLB_LIVE_GAME_URL.format(game_id=game_id),
                errors=errors,
                label=f"MLB bullpen live feed fetch for game {game_id}",
            )

            if live_payload is None:
                for team_id in sides_for_targets.values():
                    accumulators[team_id]["has_data_issue"] = True
                continue

            usage_by_side, actual_innings = _extract_game_bullpen_usage(
                live_payload,
                game_id=game_id,
                game_date=historical_date,
                errors=errors,
            )

            start_time = str(game.get("gameDate") or "")

            for side, team_id in sides_for_targets.items():
                team_usage = usage_by_side.get(side)

                if team_usage is None:
                    accumulators[team_id]["has_data_issue"] = True
                else:
                    accumulators[team_id]["pitches_by_date"][
                        historical_date
                    ] += int(team_usage["reliever_pitches"])
                    accumulators[team_id]["appearances_by_date"][
                        historical_date
                    ] += int(team_usage["reliever_appearances"])

                    for reliever in team_usage["relievers"]:
                        pitcher_id = int(reliever["pitcher_id"])
                        accumulators[team_id]["reliever_dates"][
                            pitcher_id
                        ].add(historical_date)

                if historical_date == previous_day:
                    accumulators[team_id]["previous_day_games"].append(
                        {
                            "game_id": game_id,
                            "start_time": start_time,
                            "actual_innings": actual_innings,
                        }
                    )

    rows: List[Dict[str, Any]] = []

    for team_id, info in accumulators.items():
        bullpen_data_available = (
            entire_schedule_window_available
            and not info["has_data_issue"]
        )

        completed_games_last_3d: Optional[int]
        if entire_schedule_window_available:
            completed_games_last_3d = int(info["completed_games"])
        else:
            completed_games_last_3d = None

        extra_innings_previous_game: Optional[bool] = None
        previous_day_games = info["previous_day_games"]

        if previous_day_games:
            latest_previous_day_game = sorted(
                previous_day_games,
                key=lambda item: str(item.get("start_time") or ""),
                reverse=True,
            )[0]
            actual_innings = latest_previous_day_game.get("actual_innings")

            if isinstance(actual_innings, int):
                extra_innings_previous_game = actual_innings > 9

        if bullpen_data_available:
            reliever_pitches_last_1d = int(
                info["pitches_by_date"].get(previous_day, 0)
            )
            reliever_pitches_last_3d = int(
                sum(
                    info["pitches_by_date"].get(day, 0)
                    for day in lookback_dates
                )
            )
            reliever_appearances_last_1d = int(
                info["appearances_by_date"].get(previous_day, 0)
            )
            reliever_appearances_last_3d = int(
                sum(
                    info["appearances_by_date"].get(day, 0)
                    for day in lookback_dates
                )
            )

            back_to_back_relievers_count = 0
            three_in_three_days_relievers_count = 0

            for appearance_dates in info["reliever_dates"].values():
                appeared = set(appearance_dates)

                has_back_to_back = (
                    {lookback_dates[0], lookback_dates[1]}.issubset(appeared)
                    or {
                        lookback_dates[1],
                        lookback_dates[2],
                    }.issubset(appeared)
                )
                if has_back_to_back:
                    back_to_back_relievers_count += 1

                if set(lookback_dates).issubset(appeared):
                    three_in_three_days_relievers_count += 1

            bullpen_fatigue_score = float(
                reliever_pitches_last_1d
                + 0.5
                * (
                    reliever_pitches_last_3d
                    - reliever_pitches_last_1d
                )
            )
        else:
            reliever_pitches_last_1d = None
            reliever_pitches_last_3d = None
            reliever_appearances_last_1d = None
            reliever_appearances_last_3d = None
            back_to_back_relievers_count = None
            three_in_three_days_relievers_count = None
            bullpen_fatigue_score = None

        rows.append(
            {
                "team_id": team_id,
                "team_name": info["team_name"],
                "as_of_date": as_of_date,
                "captured_at": captured_at,
                "bullpen_source": BULLPEN_SOURCE,
                "bullpen_data_available": bullpen_data_available,
                "completed_games_last_3d": completed_games_last_3d,
                "extra_innings_previous_game": extra_innings_previous_game,
                "reliever_pitches_last_1d": reliever_pitches_last_1d,
                "reliever_pitches_last_3d": reliever_pitches_last_3d,
                "reliever_appearances_last_1d": (
                    reliever_appearances_last_1d
                ),
                "reliever_appearances_last_3d": (
                    reliever_appearances_last_3d
                ),
                "back_to_back_relievers_count": (
                    back_to_back_relievers_count
                ),
                "three_in_three_days_relievers_count": (
                    three_in_three_days_relievers_count
                ),
                "closer_candidate_pitcher_id": None,
                "closer_candidate_name": None,
                "closer_identification_method": None,
                "closer_used_last_1d": None,
                "closer_pitches_last_2d": None,
                "closer_available_estimate": None,
                "closer_available_estimate_reason": (
                    "Closer candidate not identified in bullpen_context_v1."
                ),
                "bullpen_fatigue_score": bullpen_fatigue_score,
            }
        )

    return pd.DataFrame(rows, columns=BULLPEN_CONTEXT_COLUMNS)
