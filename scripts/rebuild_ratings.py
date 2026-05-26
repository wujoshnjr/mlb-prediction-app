# scripts/rebuild_ratings.py
"""Safely rebuild Elo and Glicko2 rating state from finalized MLB games.

The primary source for repair is data/finalized_games.csv, created directly
from the MLB Stats API by backfill_final_scores.py. Older feature parquet files
are only a fallback because they may not contain matchup identifiers or scores.
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

PRIMARY_FINALIZED_FILE = Path("data/finalized_games.csv")
LEGACY_CSV_FILE = Path("data/historical_predictions.csv")
LEGACY_HIST_DIR = Path("data/historical")

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
    "Sacramento Athletics": "Athletics",
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
    """Return a canonical MLB team name or None."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return TEAM_NAME_MAP.get(text, text)


def normalize_dates_to_utc(values: pd.Series) -> pd.Series:
    """Return UTC-aware datetimes for stable chronological sorting."""
    try:
        return pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        return values.map(
            lambda value: pd.to_datetime(value, errors="coerce", utc=True)
        )


def normalize_game_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize a finalized-game data source into the rebuild schema."""
    required = {"game_id", "home_team", "away_team", "home_score", "away_score"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        print(f"Skipping {source}: missing columns {missing}")
        return pd.DataFrame()

    date_column = "game_date" if "game_date" in frame.columns else (
        "date" if "date" in frame.columns else None
    )
    if date_column is None:
        print(f"Skipping {source}: missing date/game_date column")
        return pd.DataFrame()

    output = frame[
        ["game_id", date_column, "home_team", "away_team", "home_score", "away_score"]
    ].copy()
    output = output.rename(columns={date_column: "game_date"})
    output["game_id"] = output["game_id"].map(normalize_game_id)
    output["home_team"] = output["home_team"].map(normalize_team_name)
    output["away_team"] = output["away_team"].map(normalize_team_name)
    output["home_score"] = pd.to_numeric(output["home_score"], errors="coerce")
    output["away_score"] = pd.to_numeric(output["away_score"], errors="coerce")
    output["game_date"] = normalize_dates_to_utc(output["game_date"])
    output = output.dropna(
        subset=["game_id", "game_date", "home_team", "away_team", "home_score", "away_score"]
    )
    output = output[
        (output["home_team"] != output["away_team"])
        & (output["home_team"] != "")
        & (output["away_team"] != "")
    ]
    return output.reset_index(drop=True)


def merge_and_deduplicate(frames: list[pd.DataFrame], stats: dict[str, Any]) -> pd.DataFrame:
    """Merge normalized frames and remove duplicate MLB game identifiers."""
    if not frames:
        return pd.DataFrame()

    games = pd.concat(frames, ignore_index=True)
    games["game_id"] = games["game_id"].astype("string")
    games["home_team"] = games["home_team"].astype("string")
    games["away_team"] = games["away_team"].astype("string")
    games["game_date"] = normalize_dates_to_utc(games["game_date"])
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    games = games.dropna(
        subset=["game_id", "game_date", "home_team", "away_team", "home_score", "away_score"]
    )
    before = len(games)
    games = games.sort_values(["game_date", "game_id"], kind="mergesort")
    games = games.drop_duplicates(subset=["game_id"], keep="last")
    stats["duplicate_rows_removed"] = before - len(games)
    return games.reset_index(drop=True)


def read_source(path: Path, stats: dict[str, Any]) -> pd.DataFrame:
    """Read and normalize one CSV or parquet input source defensively."""
    stats["source_files_read"] += 1
    try:
        raw = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
        normalized = normalize_game_frame(raw, str(path))
        if normalized.empty:
            stats["source_files_skipped"] += 1
        return normalized
    except Exception as exc:
        stats["source_files_skipped"] += 1
        print(f"Skipping {path}: unable to read file ({exc})")
        return pd.DataFrame()


def load_historical_games() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load primary finalized history and use legacy data only as fallback."""
    stats: dict[str, Any] = {
        "primary_source": str(PRIMARY_FINALIZED_FILE),
        "primary_games_loaded": 0,
        "legacy_fallback_used": False,
        "source_files_read": 0,
        "source_files_skipped": 0,
        "duplicate_rows_removed": 0,
    }

    frames: list[pd.DataFrame] = []
    if PRIMARY_FINALIZED_FILE.exists():
        primary = read_source(PRIMARY_FINALIZED_FILE, stats)
        if not primary.empty:
            frames.append(primary)
            stats["primary_games_loaded"] = int(len(primary))

    primary_games = merge_and_deduplicate(frames, stats)
    if len(primary_games) >= MIN_GAMES:
        return primary_games, stats

    stats["legacy_fallback_used"] = True
    if LEGACY_CSV_FILE.exists():
        legacy_csv = read_source(LEGACY_CSV_FILE, stats)
        if not legacy_csv.empty:
            frames.append(legacy_csv)

    if LEGACY_HIST_DIR.exists():
        for parquet_path in sorted(LEGACY_HIST_DIR.glob("*.parquet")):
            legacy_parquet = read_source(parquet_path, stats)
            if not legacy_parquet.empty:
                frames.append(legacy_parquet)

    return merge_and_deduplicate(frames, stats), stats


def save_glicko_league(league: Glicko2League, path: Path) -> None:
    """Write Glicko2 state in the format supported by the ratings class."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(league, "save"):
        league.save(str(path))
        return
    if hasattr(league, "save_ratings"):
        league.save_ratings(str(path))
        return

    payload = {
        name: {
            "mu": float(getattr(team, "rating", 1500.0)),
            "phi": float(getattr(team, "rd", 350.0)),
            "sigma": float(getattr(team, "vol", 0.06)),
        }
        for name, team in league.teams.items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(report: dict[str, Any]) -> None:
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")


def rebuild() -> None:
    """Rebuild and validate Elo/Glicko2 rating state."""
    games, load_stats = load_historical_games()
    early_report = {"processed_games": 0, "team_count": 0, **load_stats}

    if games.empty or len(games) < MIN_GAMES:
        write_report(early_report)
        raise SystemExit(
            f"ERROR: only {len(games)} finalized games with valid scores were found; "
            f"at least {MIN_GAMES} are required. Rating files were not overwritten."
        )

    all_teams = sorted(set(games["home_team"].astype(str)).union(games["away_team"].astype(str)))
    early_report["team_count"] = len(all_teams)
    if len(all_teams) < MIN_TEAMS:
        write_report(early_report)
        raise SystemExit(
            f"ERROR: finalized history covers only {len(all_teams)} teams; "
            f"at least {MIN_TEAMS} are required. Rating files were not overwritten."
        )

    elo_ratings = {team: 1500.0 for team in all_teams}
    league = Glicko2League()
    for team in all_teams:
        league.add_team(team, rating=1500.0, rd=350.0, vol=0.06)

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
        simple_elo_update(elo_ratings, home_team, away_team, home_score, away_score)

        home_rating = league.teams[home_team]
        away_rating = league.teams[away_team]
        home_opponent = copy.deepcopy(away_rating)
        away_opponent = copy.deepcopy(home_rating)
        if home_score > away_score:
            home_result, away_result = 1.0, 0.0
        elif home_score < away_score:
            home_result, away_result = 0.0, 1.0
        else:
            home_result, away_result = 0.5, 0.5
        home_rating.update(home_opponent, home_result)
        away_rating.update(away_opponent, away_result)
        processed_ids.add(game_id)

    elo_values = [float(value) for value in elo_ratings.values()]
    glicko_values = [float(getattr(team, "rating", 1500.0)) for team in league.teams.values()]
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
    write_report(report)

    if len(processed_ids) < MIN_GAMES:
        raise SystemExit(
            f"ERROR: only {len(processed_ids)} unique finalized games were processed; "
            f"at least {MIN_GAMES} are required. Rating files were not overwritten."
        )
    if elo_range > MAX_ELO_RANGE or glicko_range > MAX_GLICKO_RANGE:
        raise SystemExit(
            "ERROR: rebuilt rating range is unsafe "
            f"(Elo={elo_range:.1f}, Glicko2={glicko_range:.1f}). "
            "Production rating files were not overwritten."
        )

    ELO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ELO_OUTPUT.write_text(json.dumps(elo_ratings, indent=2, sort_keys=True), encoding="utf-8")
    save_glicko_league(league, GLICKO_OUTPUT)
    RATED_IDS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RATED_IDS_OUTPUT.write_text(json.dumps(sorted(processed_ids), indent=2), encoding="utf-8")

    print("Rating rebuild completed successfully.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    rebuild()
