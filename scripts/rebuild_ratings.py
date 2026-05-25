# scripts/rebuild_ratings.py
"""Safely rebuild Elo and Glicko2 rating state from finalized historical games."""

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
    "Arizona Diamondbacks": "D-backs", "Diamondbacks": "D-backs", "Arizona": "D-backs",
    "Atlanta Braves": "Braves", "Atlanta": "Braves",
    "Baltimore Orioles": "Orioles", "Baltimore": "Orioles",
    "Boston Red Sox": "Red Sox", "Boston": "Red Sox",
    "Chicago Cubs": "Cubs", "Chicago (NL)": "Cubs",
    "Chicago White Sox": "White Sox", "Chicago (AL)": "White Sox",
    "Cincinnati Reds": "Reds", "Cincinnati": "Reds",
    "Cleveland Guardians": "Guardians", "Cleveland": "Guardians",
    "Colorado Rockies": "Rockies", "Colorado": "Rockies",
    "Detroit Tigers": "Tigers", "Detroit": "Tigers",
    "Houston Astros": "Astros", "Houston": "Astros",
    "Kansas City Royals": "Royals", "Kansas City": "Royals",
    "Los Angeles Angels": "Angels", "Los Angeles (AL)": "Angels",
    "Los Angeles Dodgers": "Dodgers", "Los Angeles (NL)": "Dodgers",
    "Miami Marlins": "Marlins", "Miami": "Marlins",
    "Milwaukee Brewers": "Brewers", "Milwaukee": "Brewers",
    "Minnesota Twins": "Twins", "Minnesota": "Twins",
    "New York Mets": "Mets", "New York (NL)": "Mets",
    "New York Yankees": "Yankees", "New York (AL)": "Yankees",
    "Oakland Athletics": "Athletics", "Athletics": "Athletics",
    "Philadelphia Phillies": "Phillies", "Philadelphia": "Phillies",
    "Pittsburgh Pirates": "Pirates", "Pittsburgh": "Pirates",
    "San Diego Padres": "Padres", "San Diego": "Padres",
    "San Francisco Giants": "Giants", "San Francisco": "Giants",
    "Seattle Mariners": "Mariners", "Seattle": "Mariners",
    "St. Louis Cardinals": "Cardinals", "St. Louis": "Cardinals",
    "Tampa Bay Rays": "Rays", "Tampa Bay": "Rays",
    "Texas Rangers": "Rangers", "Texas": "Rangers",
    "Toronto Blue Jays": "Blue Jays", "Toronto": "Blue Jays",
    "Washington Nationals": "Nationals", "Washington": "Nationals",
    "D-backs": "D-backs", "Braves": "Braves", "Orioles": "Orioles",
    "Red Sox": "Red Sox", "Cubs": "Cubs", "White Sox": "White Sox",
    "Reds": "Reds", "Guardians": "Guardians", "Rockies": "Rockies",
    "Tigers": "Tigers", "Astros": "Astros", "Royals": "Royals",
    "Angels": "Angels", "Dodgers": "Dodgers", "Marlins": "Marlins",
    "Brewers": "Brewers", "Twins": "Twins", "Mets": "Mets",
    "Yankees": "Yankees", "Phillies": "Phillies", "Pirates": "Pirates",
    "Padres": "Padres", "Giants": "Giants", "Mariners": "Mariners",
    "Cardinals": "Cardinals", "Rays": "Rays", "Rangers": "Rangers",
    "Blue Jays": "Blue Jays", "Nationals": "Nationals",
}


def normalize_game_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else None


def normalize_game_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    required_columns = {"game_id", "home_team", "away_team", "home_score", "away_score"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        print(f"è·³è¿ {source}: ç¼ºå°å {missing_columns}")
        return pd.DataFrame()

    date_column = None
    if "game_date" in frame.columns:
        date_column = "game_date"
    elif "date" in frame.columns:
        date_column = "date"
    else:
        print(f"è·³è¿ {source}: ç¼ºå° date/game_date")
        return pd.DataFrame()

    normalized = frame[
        ["game_id", date_column, "home_team", "away_team", "home_score", "away_score"]
    ].copy()
    normalized = normalized.rename(columns={date_column: "game_date"})
    normalized = normalized.dropna(subset=["game_id", "home_team", "away_team", "home_score", "away_score"])
    if normalized.empty:
        return normalized

    normalized["game_id"] = normalized["game_id"].map(normalize_game_id)
    normalized = normalized.dropna(subset=["game_id"])
    normalized["home_team"] = normalized["home_team"].map(TEAM_NAME_MAP).fillna(normalized["home_team"])
    normalized["away_team"] = normalized["away_team"].map(TEAM_NAME_MAP).fillna(normalized["away_team"])
    normalized["home_score"] = pd.to_numeric(normalized["home_score"], errors="coerce")
    normalized["away_score"] = pd.to_numeric(normalized["away_score"], errors="coerce")
    normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce")
    normalized = normalized.dropna(subset=["home_score", "away_score", "game_date"])
    return normalized


def load_historical_games() -> tuple[pd.DataFrame, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    stats = {"source_files_read": 0, "source_files_skipped": 0}

    if CSV_FILE.exists():
        stats["source_files_read"] += 1
        csv_frame = normalize_game_frame(pd.read_csv(CSV_FILE), str(CSV_FILE))
        if not csv_frame.empty:
            frames.append(csv_frame)
        else:
            stats["source_files_skipped"] += 1

    if HIST_DIR.exists():
        for path in sorted(HIST_DIR.glob("*.parquet")):
            stats["source_files_read"] += 1
            try:
                frame = pd.read_parquet(path)
            except Exception as exc:
                print(f"è·³è¿ {path}: æ æ³è¯»å ({exc})")
                stats["source_files_skipped"] += 1
                continue
            normalized = normalize_game_frame(frame, str(path))
            if not normalized.empty:
                frames.append(normalized)
            else:
                stats["source_files_skipped"] += 1

    if not frames:
        return pd.DataFrame(), stats

    all_games = pd.concat(frames, ignore_index=True)
    pre_dedup_count = len(all_games)
    all_games = all_games.sort_values(["game_date", "game_id"])
    all_games = all_games.drop_duplicates(subset=["game_id"], keep="last")
    stats["duplicate_rows_removed"] = pre_dedup_count - len(all_games)
    return all_games.reset_index(drop=True), stats


def serialize_glicko(league: Glicko2League, path: Path) -> None:
    # Use the league's own serializer to maintain the repository's expected schema.
    path.parent.mkdir(parents=True, exist_ok=True)
    league.save(str(path))


def rebuild() -> None:
    games, load_stats = load_historical_games()
    if games.empty or len(games) < MIN_GAMES:
        raise SystemExit(
            f"éè¯¯: åªæ {len(games)} åºå·ææ­£å¼æ¯åç final gamesï¼"
            f"å°äºæä½è¦æ± {MIN_GAMES}ï¼æç»éå»º"
        )

    all_teams = sorted(set(games["home_team"]).union(set(games["away_team"])))
    if len(all_teams) < MIN_TEAMS:
        raise SystemExit(
            f"éè¯¯: ä»è¦ç {len(all_teams)} æ¯çéï¼å°äº {MIN_TEAMS}ï¼æç»éå»º"
        )

    elo_ratings = {team: 1500.0 for team in all_teams}
    league = Glicko2League()
    for team in all_teams:
        league.add_team(team, 1500.0, 350.0, 0.06)

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
        home_opponent_snapshot = copy.deepcopy(away_rating)
        away_opponent_snapshot = copy.deepcopy(home_rating)

        if home_score > away_score:
            home_result, away_result = 1.0, 0.0
        elif home_score < away_score:
            home_result, away_result = 0.0, 1.0
        else:
            home_result = away_result = 0.5

        home_rating.update(home_opponent_snapshot, home_result)
        away_rating.update(away_opponent_snapshot, away_result)
        processed_ids.add(game_id)

    elo_values = list(elo_ratings.values())
    glicko_values = [team.rating for team in league.teams.values()]
    elo_range = max(elo_values) - min(elo_values)
    glicko_range = max(glicko_values) - min(glicko_values)

    report = {
        "processed": len(processed_ids),
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
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if elo_range > MAX_ELO_RANGE or glicko_range > MAX_GLICKO_RANGE:
        raise SystemExit(
            "éè¯¯: éå»ºç»æ rating èå´å¼å¸¸ "
            f"(Elo={elo_range:.1f}, Glicko2={glicko_range:.1f})ï¼"
            "å·²ä¿å­æ¥åï¼ä½æç»è¦çæ­£å¼ rating state"
        )

    ELO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ELO_OUTPUT.write_text(json.dumps(elo_ratings, indent=2), encoding="utf-8")
    serialize_glicko(league, GLICKO_OUTPUT)
    RATED_IDS_OUTPUT.write_text(
        json.dumps(sorted(processed_ids), indent=2),
        encoding="utf-8",
    )

    print("â Rating éå»ºæå")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    rebuild()
