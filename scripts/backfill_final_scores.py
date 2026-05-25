# scripts/backfill_final_scores.py
"""Backfill finalized MLB scores into historical CSV and parquet datasets.

Only games confirmed as Final by the MLB Stats API are written. Non-final games
and API failures are counted separately and never overwrite existing records.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

HISTORY_FILE = Path("data/historical_predictions.csv")
HIST_DIR = Path("data/historical")
REQUEST_TIMEOUT_SECONDS = 10
REQUEST_SLEEP_SECONDS = 0.30


@dataclass(frozen=True)
class ScoreFetchResult:
    status: str
    home_win: int | None = None
    home_score: int | None = None
    away_score: int | None = None
    detail: str = ""


def normalize_game_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else None


def fetch_final_score(game_id: str) -> ScoreFetchResult:
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return ScoreFetchResult("api_error", detail=str(exc))

    status = data.get("gameData", {}).get("status", {})
    abstract_state = str(status.get("abstractGameState", ""))
    detailed_state = str(status.get("detailedState", ""))

    if abstract_state != "Final":
        return ScoreFetchResult(
            "non_final",
            detail=f"{abstract_state or 'Unknown'} / {detailed_state or 'Unknown'}",
        )

    teams = data.get("liveData", {}).get("linescore", {}).get("teams", {})
    home_runs = teams.get("home", {}).get("runs")
    away_runs = teams.get("away", {}).get("runs")
    if home_runs is None or away_runs is None:
        return ScoreFetchResult("missing_score", detail="Final game missing linescore runs")

    home_score = int(home_runs)
    away_score = int(away_runs)
    home_win = 1 if home_score > away_score else 0
    return ScoreFetchResult(
        "final_success",
        home_win=home_win,
        home_score=home_score,
        away_score=away_score,
    )


def ensure_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ("home_win", "home_score", "away_score"):
        if column not in output.columns:
            output[column] = np.nan
    return output


def update_frame(
    frame: pd.DataFrame,
    already_fetched: dict[str, ScoreFetchResult],
    stats: dict[str, int],
) -> tuple[pd.DataFrame, bool]:
    if "game_id" not in frame.columns:
        print("è·³éç¼ºå° game_id çè³æè¡¨")
        stats["missing_game_id_column"] += 1
        return frame, False

    output = ensure_score_columns(frame)
    missing_mask = (
        output["game_id"].notna()
        & (output["home_score"].isna() | output["away_score"].isna())
    )
    changed = False

    for index, row in output[missing_mask].iterrows():
        game_id = normalize_game_id(row.get("game_id"))
        if not game_id:
            stats["invalid_game_id"] += 1
            continue

        if game_id in already_fetched:
            stats["duplicate_rows"] += 1
            result = already_fetched[game_id]
        else:
            stats["unique_games_requested"] += 1
            result = fetch_final_score(game_id)
            already_fetched[game_id] = result
            time.sleep(REQUEST_SLEEP_SECONDS)

            if result.status in stats:
                stats[result.status] += 1

        stats["rows_scanned"] += 1
        if result.status != "final_success":
            continue

        output.at[index, "home_win"] = result.home_win
        output.at[index, "home_score"] = result.home_score
        output.at[index, "away_score"] = result.away_score
        stats["rows_backfilled"] += 1
        changed = True

    return output, changed


def backfill() -> None:
    stats = {
        "rows_scanned": 0,
        "rows_backfilled": 0,
        "unique_games_requested": 0,
        "final_success": 0,
        "non_final": 0,
        "api_error": 0,
        "missing_score": 0,
        "duplicate_rows": 0,
        "invalid_game_id": 0,
        "missing_game_id_column": 0,
        "files_updated": 0,
    }
    already_fetched: dict[str, ScoreFetchResult] = {}

    if HISTORY_FILE.exists():
        csv_frame = pd.read_csv(HISTORY_FILE)
        csv_frame, changed = update_frame(csv_frame, already_fetched, stats)
        if changed:
            csv_frame.to_csv(HISTORY_FILE, index=False)
            stats["files_updated"] += 1
            print(f"å·²æ´æ° {HISTORY_FILE}")

    if HIST_DIR.exists():
        for parquet_path in sorted(HIST_DIR.glob("*.parquet")):
            try:
                parquet_frame = pd.read_parquet(parquet_path)
            except Exception as exc:
                stats["api_error"] += 1
                print(f"è®å {parquet_path} å¤±æ: {exc}")
                continue

            parquet_frame, changed = update_frame(parquet_frame, already_fetched, stats)
            if changed:
                parquet_frame.to_parquet(parquet_path, index=False)
                stats["files_updated"] += 1
                print(f"å·²æ´æ° {parquet_path}")

    print("===== Backfill Summary =====")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    backfill()
