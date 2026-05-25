# scripts/rating_updater.py
"""Update Elo or Glicko2 ratings from newly finalized MLB games.

This module is used by:
- prediction.py to load Glicko2 rating state.
- rebuild_ratings.py to reuse the Elo update formula.
- GitHub Actions for incremental daily rating updates.

Runtime text is ASCII-only to avoid encoding damage during browser edits.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.glicko2_ratings import Glicko2League

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ELO_FILE = Path("data/elo_ratings.json")
GLICKO_FILE = Path("data/glicko2_ratings.json")
RATED_GAMES_FILE = Path("data/rated_game_ids.json")
FINAL_RESULTS_FILE = Path("data/new_final_results.json")

DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOL = 0.06
DEFAULT_ELO_K = 32.0
DEFAULT_HOME_ADVANTAGE = 24.0


def normalize_game_id(value: Any) -> str | None:
    """Return a stable string game id, or None when the value is invalid."""
    if value is None:
        return None

    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None

    return text


def load_elo_ratings() -> dict[str, float]:
    """Load Elo rating state from disk."""
    if not ELO_FILE.exists():
        return {}

    try:
        payload = json.loads(ELO_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.error("Unable to read %s: %s", ELO_FILE, exc)
        return {}

    if not isinstance(payload, dict):
        LOGGER.error("%s does not contain a JSON object.", ELO_FILE)
        return {}

    ratings: dict[str, float] = {}
    for team, value in payload.items():
        try:
            ratings[str(team)] = float(value)
        except (TypeError, ValueError):
            LOGGER.warning("Skipping non-numeric Elo value for team %s.", team)

    return ratings


def save_elo_ratings(ratings: dict[str, float]) -> None:
    """Write Elo rating state to disk."""
    ELO_FILE.parent.mkdir(parents=True, exist_ok=True)
    ELO_FILE.write_text(
        json.dumps(ratings, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_glicko2_league() -> Glicko2League:
    """Load Glicko2 state, or initialize it from current Elo state."""
    if GLICKO_FILE.exists():
        return Glicko2League.load(str(GLICKO_FILE))

    league = Glicko2League()
    elo_ratings = load_elo_ratings()

    for team_id, elo_value in elo_ratings.items():
        adjusted_rating = float(elo_value) - DEFAULT_HOME_ADVANTAGE
        league.add_team(
            team_id,
            rating=adjusted_rating,
            rd=DEFAULT_RD,
            vol=DEFAULT_VOL,
        )

    return league


def save_glicko2_league(league: Glicko2League) -> None:
    """Write Glicko2 rating state to disk."""
    GLICKO_FILE.parent.mkdir(parents=True, exist_ok=True)
    league.save(str(GLICKO_FILE))


def load_rated_game_ids() -> set[str]:
    """Load game ids that have already affected rating state."""
    if not RATED_GAMES_FILE.exists():
        return set()

    try:
        payload = json.loads(RATED_GAMES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.error("Unable to read %s: %s", RATED_GAMES_FILE, exc)
        return set()

    if not isinstance(payload, list):
        LOGGER.error("%s does not contain a JSON list.", RATED_GAMES_FILE)
        return set()

    return {
        game_id
        for value in payload
        if (game_id := normalize_game_id(value)) is not None
    }


def save_rated_game_ids(game_ids: set[str]) -> None:
    """Write processed rating game ids to disk."""
    RATED_GAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    RATED_GAMES_FILE.write_text(
        json.dumps(sorted(game_ids), indent=2),
        encoding="utf-8",
    )


def simple_elo_update(
    elo_dict: dict[str, float],
    home_team: str,
    away_team: str,
    home_score: int | float,
    away_score: int | float,
    k_factor: float = DEFAULT_ELO_K,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> dict[str, float]:
    """Update two Elo ratings from one final game result."""
    home_rating = float(elo_dict.get(home_team, DEFAULT_RATING))
    away_rating = float(elo_dict.get(away_team, DEFAULT_RATING))

    expected_home = 1.0 / (
        1.0
        + 10.0
        ** ((away_rating - (home_rating + home_advantage)) / 400.0)
    )

    if home_score > away_score:
        actual_home = 1.0
    elif home_score < away_score:
        actual_home = 0.0
    else:
        actual_home = 0.5

    adjustment = k_factor * (actual_home - expected_home)
    elo_dict[home_team] = home_rating + adjustment
    elo_dict[away_team] = away_rating - adjustment

    return elo_dict


def filter_new_games(
    game_results: list[dict[str, Any]],
    rated_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return unprocessed finalized games and duplicate ids."""
    new_games: list[dict[str, Any]] = []
    skipped_ids: list[str] = []

    for game in game_results:
        if not isinstance(game, dict):
            LOGGER.warning("Skipping malformed final game payload: %r", game)
            continue

        game_id = normalize_game_id(game.get("game_id"))
        if game_id is None:
            LOGGER.warning("Skipping final game payload without a valid game_id.")
            continue

        if game_id in rated_ids:
            skipped_ids.append(game_id)
            continue

        required_fields = ("home_team", "away_team", "home_score", "away_score")
        missing_fields = [
            field for field in required_fields if game.get(field) is None
        ]
        if missing_fields:
            LOGGER.warning(
                "Skipping game %s because fields are missing: %s",
                game_id,
                missing_fields,
            )
            continue

        normalized_game = dict(game)
        normalized_game["game_id"] = game_id
        normalized_game["home_team"] = str(game["home_team"])
        normalized_game["away_team"] = str(game["away_team"])

        try:
            normalized_game["home_score"] = int(game["home_score"])
            normalized_game["away_score"] = int(game["away_score"])
        except (TypeError, ValueError):
            LOGGER.warning("Skipping game %s because score values are invalid.", game_id)
            continue

        new_games.append(normalized_game)

    return new_games, skipped_ids


def update_elo_ratings(new_games: list[dict[str, Any]]) -> None:
    """Apply incremental Elo updates."""
    ratings = load_elo_ratings()

    for game in new_games:
        simple_elo_update(
            ratings,
            game["home_team"],
            game["away_team"],
            game["home_score"],
            game["away_score"],
        )

    save_elo_ratings(ratings)


def update_glicko2_ratings(new_games: list[dict[str, Any]]) -> None:
    """Apply incremental Glicko2 updates using opponent snapshots."""
    league = load_glicko2_league()

    for game in new_games:
        home_team_id = game["home_team"]
        away_team_id = game["away_team"]

        if home_team_id not in league.teams:
            league.add_team(home_team_id)
        if away_team_id not in league.teams:
            league.add_team(away_team_id)

        home_team = league.teams[home_team_id]
        away_team = league.teams[away_team_id]

        home_opponent_snapshot = copy.deepcopy(away_team)
        away_opponent_snapshot = copy.deepcopy(home_team)

        if game["home_score"] > game["away_score"]:
            home_result = 1.0
            away_result = 0.0
        elif game["home_score"] < game["away_score"]:
            home_result = 0.0
            away_result = 1.0
        else:
            home_result = 0.5
            away_result = 0.5

        home_team.update(home_opponent_snapshot, home_result)
        away_team.update(away_opponent_snapshot, away_result)

    save_glicko2_league(league)


def update_ratings(game_results: list[dict[str, Any]]) -> None:
    """Update configured rating engine once for each previously unseen game."""
    rated_ids = load_rated_game_ids()
    new_games, skipped_ids = filter_new_games(game_results, rated_ids)

    if skipped_ids:
        LOGGER.info(
            "Skipped %d previously rated games.",
            len(skipped_ids),
        )

    if not new_games:
        LOGGER.info("No new finalized games require rating updates.")
        return

    LOGGER.info("Updating ratings from %d new finalized games.", len(new_games))

    rating_engine = str(getattr(config, "RATINGS_ENGINE", "elo")).lower()

    if rating_engine == "elo":
        update_elo_ratings(new_games)
    elif rating_engine == "glicko2":
        update_glicko2_ratings(new_games)
    else:
        raise ValueError(f"Unknown rating engine: {rating_engine}")

    for game in new_games:
        rated_ids.add(game["game_id"])

    save_rated_game_ids(rated_ids)

    LOGGER.info(
        "Rating update complete. new_games=%d skipped_duplicates=%d rated_game_ids=%d",
        len(new_games),
        len(skipped_ids),
        len(rated_ids),
    )


def load_final_results() -> list[dict[str, Any]]:
    """Load newly finalized games produced by update_results.py."""
    if not FINAL_RESULTS_FILE.exists():
        LOGGER.warning(
            "%s does not exist. Run update_results.py before rating update.",
            FINAL_RESULTS_FILE,
        )
        return []

    try:
        payload = json.loads(FINAL_RESULTS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.error("Unable to read %s: %s", FINAL_RESULTS_FILE, exc)
        return []

    if not isinstance(payload, list):
        LOGGER.error("%s does not contain a JSON list.", FINAL_RESULTS_FILE)
        return []

    return payload


def main() -> None:
    game_results = load_final_results()

    if not game_results:
        LOGGER.info("No finalized games were provided for incremental rating update.")
        return

    update_ratings(game_results)


if __name__ == "__main__":
    main()
