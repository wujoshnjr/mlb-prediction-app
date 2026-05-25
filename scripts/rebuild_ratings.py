# scripts/rebuild_ratings.py
"""Safely rebuild Elo and Glicko2 rating state from finalized historical games.

This repair script:
- Reads finalized historical games from CSV and parquet files.
- Normalizes team names and game identifiers.
- Removes duplicate games before rebuilding ratings.
- Forces sortable pandas dtypes after concatenating heterogeneous data files.
- Writes production rating files only when safety checks pass.

Runtime text is ASCII-only to avoid encoding damage during browser edits.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.glicko2_ratings import Glicko2League
from scripts.rating_updater import simple_elo_update

CSV_FILE = Path("data/historical_predictions.csv")
HIST_DIR = Path("data/historical")

ELO_OUTPUT = Path("data/elo_ratings.json")
GLICKO_OUTPUT = Path("data/glicko2_ratings.json")
RATED_IDS_OUTPUT = Path("data/rated_game_ids.json")
REPORT_OUTPUT = Path("report/rating_rebuild_report.json")

MIN_GAMES = 100
MIN_TEAMS = 30
MAX_ELO_RANGE = 400.0
MAX_GLICKO_RANGE = 400.0

TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "D-backs",
    "Diamondbacks": "D-backs",
    "Arizona": "D-backs",
    "Atlanta Braves": "Braves",
    "Atlanta": "Braves",
    "Baltimore Orioles": "Orioles",
    "Baltimore": "Orioles",
    "Boston Red Sox": "Red Sox",
    "Boston": "Red Sox",
    "Chicago Cubs": "Cubs",
    "Chicago (NL)": "Cubs",
    "Chicago White Sox": "White Sox",
    "Chicago (AL)": "White Sox",
    "Cincinnati Reds": "Reds",
    "Cincinnati": "Reds",
    "Cleveland Guardians": "Guardians",
    "Cleveland": "Guardians",
    "Colorado Rockies": "Rockies",
    "Colorado": "Rockies",
    "Detroit Tigers": "Tigers",
    "Detroit": "Tigers",
    "Houston Astros": "Astros",
    "Houston": "Astros",
    "Kansas City Royals": "Royals",
    "Kansas City": "Royals",
    "Los Angeles Angels": "Angels",
    "Los Angeles (AL)": "Angels",
    "Los Angeles Dodgers": "Dodgers",
    "Los Angeles (NL)": "Dodgers",
    "Miami Marlins": "Marlins",
    "Miami": "Marlins",
    "Milwaukee Brewers": "Brewers",
    "Milwaukee": "Brewers",
    "Minnesota Twins": "Twins",
    "Minnesota": "Twins",
    "New York Mets": "Mets",
    "New York (NL)": "Mets",
    "New York Yankees": "Yankees",
    "New York (AL)": "Yankees",
    "Oakland Athletics": "Athletics",
    "Oakland": "Athletics",
    "Athletics": "Athletics",
    "Philadelphia Phillies": "Phillies",
    "Philadelphia": "Phillies",
    "Pittsburgh Pirates": "Pirates",
    "Pittsburgh": "Pirates",
    "San Diego Padres": "Padres",
    "San Diego": "Padres",
    "San Francisco Giants": "Giants",
    "San Francisco": "Giants",
    "Seattle Mariners": "Mariners",
    "Seattle": "Mariners",
    "St. Louis Cardinals": "Cardinals",
    "St. Louis": "Cardinals",
    "Tampa Bay Rays": "Rays",
    "Tampa Bay": "Rays",
    "Texas Rangers": "Rangers",
    "Texas": "Rangers",
    "Toronto Blue Jays": "Blue Jays",
    "Toronto": "Blue Jays",
    "Washington Nationals": "Nationals",
    "Washington": "Nationals",
    "D-backs": "D-backs",
    "Braves": "Braves",
    "Orioles": "Orioles",
    "Red Sox": "Red Sox",
    "Cubs": "Cubs",
    "White Sox": "White Sox",
    "Reds": "Reds",
    "Guardians": "Guardians",
    "Rockies": "Rockies",
    "Tigers": "Tigers",
    "Astros": "Astros",
    "Royals": "Royals",
    "Angels": "Angels",
    "Dodgers": "Dodgers",
    "Marlins": "Marlins",
    "Brewers": "Brewers",
    "Twins": "Twins",
    "Mets": "Mets",
    "Yankees": "Yankees",
    "Phillies": "Phillies",
    "Pirates": "Pirates",
    "Padres": "Padres",
    "Giants": "Giants",
    "Mariners": "Mariners",
    "Cardinals": "Cardinals",
    "Rays": "Rays",
    "Rangers": "Rangers",
    "Blue Jays": "Blue Jays",
    "Nationals": "Nationals",
}


def normalize_game_id(value: Any) -> str | None:
    """Return a stable game identifier string or None."""
    if value is None or pd.isna(value):
        return None

    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None

    return text


def normalize_team_name(value: Any) -> str | None:
    """Return a canonical MLB team name or None for invalid input."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None

    return TEAM_NAME_MAP.get(text, text)


def normalize_game_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize one historical data source into the rebuild schema."""
    required_columns = {
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }

    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        print(f"Skipping {source}: missing columns {missing_columns}")
        return pd.DataFrame()

    if "game_date" in frame.columns:
        date_column = "game_date"
    elif "date" in frame.columns:
        date_column = "date"
    else:
        print(f"Skipping {source}: missing date/game_date column")
        return pd.DataFrame()

    normalized = frame[
        [
            "game_id",
            date_column,
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].copy()

    normalized = normalized.rename(columns={date_column: "game_date"})

    normalized["game_id"] = normalized["game_id"].map(normalize_game_id)
    normalized["home_team"] = normalized["home_team"].map(normalize_team_name)
    normalized["away_team"] = normalized["away_team"].map(normalize_team_name)

    normalized["home_score"] = pd.to_numeric(
        normalized["home_score"],
        errors="coerce",
    )
    normalized["away_score"] = pd.to_numeric(
        normalized["away_score"],
        errors="coerce",
    )
    normalized["game_date"] = pd.to_datetime(
        normalized["game_date"],
        errors="coerce",
    )

    normalized = normalized.dropna(
        subset=[
            "game_id",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "game_date",
        ]
    )

    normalized = normalized[
        (normalized["home_team"] != "")
        & (normalized["away_team"] != "")
        & (normalized["home_team"] != normalized["away_team"])
    ]

    return normalized.reset_index(drop=True)


def load_historical_games() -> tuple[pd.DataFrame, dict[str, int]]:
    """Load, normalize, merge and safely sort finalized historical games."""
    frames: list[pd.DataFrame] = []

    stats = {
        "source_files_read": 0,
        "source_files_skipped": 0,
        "duplicate_rows_removed": 0,
    }

    if CSV_FILE.exists():
        stats["source_files_read"] += 1

        try:
            csv_source = pd.read_csv(CSV_FILE)
            csv_frame = normalize_game_frame(csv_source, str(CSV_FILE))

            if not csv_frame.empty:
                frames.append(csv_frame)
            else:
                stats["source_files_skipped"] += 1
        except Exception as exc:
            stats["source_files_skipped"] += 1
            print(f"Skipping {CSV_FILE}: unable to read file ({exc})")

    if HIST_DIR.exists():
        for parquet_path in sorted(HIST_DIR.glob("*.parquet")):
            stats["source_files_read"] += 1

            try:
                parquet_source = pd.read_parquet(parquet_path)
                parquet_frame = normalize_game_frame(
                    parquet_source,
                    str(parquet_path),
                )

                if not parquet_frame.empty:
                    frames.append(parquet_frame)
                else:
                    stats["source_files_skipped"] += 1
            except Exception as exc:
                stats["source_files_skipped"] += 1
                print(f"Skipping {parquet_path}: unable to read file ({exc})")

    if not frames:
        return pd.DataFrame(), stats

    all_games = pd.concat(frames, ignore_index=True)

    # Critical repair:
    # Some parquet sources preserve game_id or team columns as unordered
    # categorical values. Pandas cannot sort unordered categorical columns.
    # Convert merged columns into stable sortable dtypes before sorting.
    all_games["game_id"] = all_games["game_id"].astype("string")
    all_games["home_team"] = all_games["home_team"].astype("string")
    all_games["away_team"] = all_games["away_team"].astype("string")

    all_games["game_date"] = pd.to_datetime(
        all_games["game_date"],
        errors="coerce",
    )
    all_games["home_score"] = pd.to_numeric(
        all_games["home_score"],
        errors="coerce",
    )
    all_games["away_score"] = pd.to_numeric(
        all_games["away_score"],
        errors="coerce",
    )

    all_games = all_games.dropna(
        subset=[
            "game_id",
            "home_team",
            "away_team",
            "game_date",
            "home_score",
            "away_score",
        ]
    )

    all_games = all_games[
        (all_games["home_team"] != "")
        & (all_games["away_team"] != "")
        & (all_games["home_team"] != all_games["away_team"])
    ]

    pre_dedup_count = len(all_games)

    all_games = all_games.sort_values(
        ["game_date", "game_id"],
        kind="mergesort",
    )
    all_games = all_games.drop_duplicates(
        subset=["game_id"],
        keep="last",
    )

    stats["duplicate_rows_removed"] = pre_dedup_count - len(all_games)

    return all_games.reset_index(drop=True), stats


def save_glicko_league(league: Glicko2League, path: Path) -> None:
    """Write Glicko2 rating state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(league, "save"):
        league.save(str(path))
        return

    if hasattr(league, "save_ratings"):
        league.save_ratings(str(path))
        return

    teams_payload: dict[str, dict[str, float]] = {}

    for name, team in league.teams.items():
        teams_payload[name] = {
            "mu": float(getattr(team, "rating", 1500.0)),
            "phi": float(getattr(team, "rd", 350.0)),
            "sigma": float(getattr(team, "vol", 0.06)),
        }

    path.write_text(
        json.dumps(teams_payload, indent=2),
        encoding="utf-8",
    )


def rebuild() -> None:
    """Rebuild and validate Elo/Glicko2 rating state."""
    games, load_stats = load_historical_games()

    if games.empty or len(games) < MIN_GAMES:
        raise SystemExit(
            f"ERROR: only {len(games)} finalized games with valid scores were "
            f"found; at least {MIN_GAMES} are required. "
            "Rating files were not overwritten."
        )

    all_teams = sorted(
        set(games["home_team"].astype(str)).union(
            set(games["away_team"].astype(str))
        )
    )

    if len(all_teams) < MIN_TEAMS:
        raise SystemExit(
            f"ERROR: historical data covers only {len(all_teams)} teams; "
            f"at least {MIN_TEAMS} are required. "
            "Rating files were not overwritten."
        )

    elo_ratings = {team: 1500.0 for team in all_teams}

    league = Glicko2League()
    for team in all_teams:
        league.add_team(
            team,
            rating=1500.0,
            rd=350.0,
            vol=0.06,
        )

    processed_ids: set[str] = set()
    skipped_invalid_score = 0

    for _, row in games.iterrows():
        game_id = normalize_game_id(row["game_id"])

        if not game_id or game_id in processed_ids:
            continue

        try:
            home_score = int(row["home_score"])
            away_score = int(row["away_score"])
        except (TypeError, ValueError):
            skipped_invalid_score += 1
            continue

        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        if home_team not in elo_ratings:
            elo_ratings[home_team] = 1500.0
        if away_team not in elo_ratings:
            elo_ratings[away_team] = 1500.0

        simple_elo_update(
            elo_ratings,
            home_team,
            away_team,
            home_score,
            away_score,
        )

        if home_team not in league.teams:
            league.add_team(home_team, rating=1500.0, rd=350.0, vol=0.06)
        if away_team not in league.teams:
            league.add_team(away_team, rating=1500.0, rd=350.0, vol=0.06)

        home_rating = league.teams[home_team]
        away_rating = league.teams[away_team]

        home_opponent_snapshot = copy.deepcopy(away_rating)
        away_opponent_snapshot = copy.deepcopy(home_rating)

        if home_score > away_score:
            home_result = 1.0
            away_result = 0.0
        elif home_score < away_score:
            home_result = 0.0
            away_result = 1.0
        else:
            home_result = 0.5
            away_result = 0.5

        home_rating.update(home_opponent_snapshot, home_result)
        away_rating.update(away_opponent_snapshot, away_result)

        processed_ids.add(game_id)

    if len(processed_ids) < MIN_GAMES:
        raise SystemExit(
            f"ERROR: only {len(processed_ids)} unique finalized games were "
            f"processed after de-duplication; at least {MIN_GAMES} are required. "
            "Rating files were not overwritten."
        )

    elo_values = [float(value) for value in elo_ratings.values()]
    glicko_values = [
        float(getattr(team, "rating", 1500.0))
        for team in league.teams.values()
    ]

    elo_range = max(elo_values) - min(elo_values)
    glicko_range = max(glicko_values) - min(glicko_values)

    report = {
        "processed_games": len(processed_ids),
        "skipped_invalid_score": skipped_invalid_score,
        "team_count": len(all_teams),
        "elo_min": min(elo_values),
        "elo_max": max(elo_values),
        "elo_range": elo_range,
        "glicko_min": min(glicko_values),
        "glicko_max": max(glicko_values),
        "glicko_range": glicko_range,
        "teams": all_teams,
        **load_stats,
    }

    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if elo_range > MAX_ELO_RANGE or glicko_range > MAX_GLICKO_RANGE:
        raise SystemExit(
            "ERROR: rebuilt rating range is unsafe "
            f"(Elo={elo_range:.1f}, Glicko2={glicko_range:.1f}). "
            "A report was written, but production rating files were not overwritten."
        )

    ELO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ELO_OUTPUT.write_text(
        json.dumps(elo_ratings, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    save_glicko_league(league, GLICKO_OUTPUT)

    RATED_IDS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RATED_IDS_OUTPUT.write_text(
        json.dumps(sorted(processed_ids), indent=2),
        encoding="utf-8",
    )

    print("Rating rebuild completed successfully.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    rebuild()
