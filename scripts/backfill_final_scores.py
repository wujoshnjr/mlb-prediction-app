# scripts/backfill_final_scores.py
"""Build a normalized finalized-games history directly from the MLB Stats API.

Older feature parquet files are not guaranteed to contain matchup identifiers or
scores. Rating repair must therefore never depend on those files being usable.
This script retrieves finalized regular-season games directly from MLB's
schedule endpoint and writes a stable dataset consumed by rebuild_ratings.py.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

FINALIZED_GAMES_FILE = Path("data/finalized_games.csv")
REPORT_FILE = Path("report/final_score_backfill_report.json")

API_URL = "https://statsapi.mlb.com/api/v1/schedule"
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_SLEEP_SECONDS = 0.10
CHUNK_DAYS = 13

# This bounded completed regular-season window contains substantially more than
# the 100 games required for a safe rating rebuild and all 30 MLB clubs.
START_DATE = date(2025, 8, 1)
END_DATE = date(2025, 9, 30)
MIN_REQUIRED_FINAL_GAMES = 100


def iter_date_windows(start_date: date, end_date: date):
    """Yield bounded inclusive date windows for API requests."""
    current = start_date
    while current <= end_date:
        window_end = min(current + timedelta(days=CHUNK_DAYS), end_date)
        yield current, window_end
        current = window_end + timedelta(days=1)


def normalize_game_id(value: Any) -> str | None:
    """Return a stable MLB game id or None."""
    if value is None or pd.isna(value):
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def fetch_window(
    start_date: date,
    end_date: date,
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fetch completed regular-season games for one inclusive window."""
    params = {
        "sportId": 1,
        "gameTypes": "R",
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
    }

    stats["api_windows_requested"] += 1
    try:
        response = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        stats["api_windows_failed"] += 1
        stats["api_errors"].append(
            f"{start_date.isoformat()}..{end_date.isoformat()}: {exc}"
        )
        return []

    rows: list[dict[str, Any]] = []
    for date_payload in payload.get("dates", []):
        official_date = str(date_payload.get("date", "")).strip()
        for game in date_payload.get("games", []):
            stats["schedule_games_seen"] += 1
            if str(game.get("gameType", "")) != "R":
                stats["non_regular_games_skipped"] += 1
                continue

            status = game.get("status", {})
            abstract_state = str(status.get("abstractGameState", ""))
            coded_state = str(status.get("codedGameState", ""))
            if abstract_state != "Final" and coded_state != "F":
                stats["non_final_games_skipped"] += 1
                continue

            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            game_id = normalize_game_id(game.get("gamePk"))
            home_name = str(home.get("team", {}).get("name", "")).strip()
            away_name = str(away.get("team", {}).get("name", "")).strip()
            home_score = home.get("score")
            away_score = away.get("score")

            if (
                not game_id
                or not official_date
                or not home_name
                or not away_name
                or home_score is None
                or away_score is None
            ):
                stats["final_games_missing_required_fields"] += 1
                continue

            rows.append(
                {
                    "game_id": game_id,
                    "game_date": official_date,
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                    "home_win": 1 if int(home_score) > int(away_score) else 0,
                    "status": "Final",
                }
            )

    stats["final_rows_received"] += len(rows)
    return rows


def write_report(stats: dict[str, Any]) -> None:
    """Write a JSON diagnostic report for workflow artifacts."""
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def build_finalized_games_history() -> None:
    """Retrieve, validate and persist the normalized finalized-game dataset."""
    stats: dict[str, Any] = {
        "source": "MLB Stats API schedule",
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "minimum_required_final_games": MIN_REQUIRED_FINAL_GAMES,
        "api_windows_requested": 0,
        "api_windows_failed": 0,
        "api_errors": [],
        "schedule_games_seen": 0,
        "non_regular_games_skipped": 0,
        "non_final_games_skipped": 0,
        "final_games_missing_required_fields": 0,
        "final_rows_received": 0,
        "duplicate_rows_removed": 0,
        "final_games_written": 0,
    }

    rows: list[dict[str, Any]] = []
    for window_start, window_end in iter_date_windows(START_DATE, END_DATE):
        rows.extend(fetch_window(window_start, window_end, stats))
        if REQUEST_SLEEP_SECONDS:
            time.sleep(REQUEST_SLEEP_SECONDS)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["game_id"] = frame["game_id"].astype("string")
        frame["game_date"] = pd.to_datetime(
            frame["game_date"], errors="coerce", utc=True
        )
        frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
        frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
        frame = frame.dropna(
            subset=[
                "game_id",
                "game_date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
            ]
        )
        pre_dedup = len(frame)
        frame = frame.sort_values(["game_date", "game_id"], kind="mergesort")
        frame = frame.drop_duplicates(subset=["game_id"], keep="last")
        stats["duplicate_rows_removed"] = pre_dedup - len(frame)

    stats["final_games_written"] = int(len(frame))
    write_report(stats)

    if len(frame) < MIN_REQUIRED_FINAL_GAMES:
        raise SystemExit(
            "ERROR: MLB finalized-games history retrieval produced only "
            f"{len(frame)} usable games; at least {MIN_REQUIRED_FINAL_GAMES} "
            "are required. Existing rating files were not modified."
        )

    output = frame.copy()
    output["game_date"] = output["game_date"].dt.strftime("%Y-%m-%d")
    FINALIZED_GAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(FINALIZED_GAMES_FILE, index=False, encoding="utf-8")

    print(
        "Finalized games history built successfully: "
        f"{len(output)} games written to {FINALIZED_GAMES_FILE}."
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    build_finalized_games_history()
