# scripts/backfill_final_scores.py
"""Backfill finalized MLB scores into historical CSV and parquet datasets.

Only games confirmed as Final by the MLB Stats API are written. Non-final games
and API failures are tracked separately and never overwrite existing records.

The repair run is intentionally bounded so a workflow_dispatch job does not
scan multiple historical seasons without a stop condition.

Runtime text is ASCII-only to avoid encoding damage during browser edits.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

HISTORY_FILE = Path("data/historical_predictions.csv")
HISTORICAL_DIR = Path("data/historical")
REPORT_FILE = Path("report/final_score_backfill_report.json")

REQUEST_TIMEOUT_SECONDS = 10
REQUEST_SLEEP_SECONDS = 0.05

# Repair only recent historical files that actually exist in this repository.
# The current historical parquet data includes 2025 files, so using a 2026
# cutoff would incorrectly skip all usable repair sources.
MINIMUM_BACKFILL_DATE = pd.Timestamp("2025-08-01")

# Enough finalized games to cover all MLB teams while remaining bounded.
TARGET_FINAL_GAMES = 500

# Used only when the CSV does not contain a usable date column.
MAX_UNDATED_CSV_ROWS = 1000


@dataclass(frozen=True)
class ScoreFetchResult:
    """Result returned after requesting one MLB game final score."""

    status: str
    home_win: int | None = None
    home_score: int | None = None
    away_score: int | None = None
    detail: str = ""


def normalize_game_id(value: Any) -> str | None:
    """Return a stable string game id or None for invalid values."""
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


def fetch_final_score(game_id: str) -> ScoreFetchResult:
    """Fetch a result only when MLB confirms the game is Final."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        return ScoreFetchResult(
            status="api_error",
            detail=str(exc),
        )

    game_status = payload.get("gameData", {}).get("status", {})
    abstract_state = str(game_status.get("abstractGameState", ""))
    detailed_state = str(game_status.get("detailedState", ""))

    if abstract_state != "Final":
        return ScoreFetchResult(
            status="non_final",
            detail=f"{abstract_state or 'Unknown'} / {detailed_state or 'Unknown'}",
        )

    teams = payload.get("liveData", {}).get("linescore", {}).get("teams", {})
    home_runs = teams.get("home", {}).get("runs")
    away_runs = teams.get("away", {}).get("runs")

    if home_runs is None or away_runs is None:
        return ScoreFetchResult(
            status="missing_score",
            detail="Final game is missing linescore runs.",
        )

    home_score = int(home_runs)
    away_score = int(away_runs)

    return ScoreFetchResult(
        status="final_success",
        home_win=1 if home_score > away_score else 0,
        home_score=home_score,
        away_score=away_score,
    )


def ensure_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure output columns needed by rating rebuild exist."""
    output = frame.copy()

    for column in ("home_win", "home_score", "away_score"):
        if column not in output.columns:
            output[column] = np.nan

    return output


def frame_missing_score_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows that have a game id but still need final scores."""
    if "game_id" not in frame.columns:
        return pd.Series(False, index=frame.index)

    return (
        frame["game_id"].notna()
        & (
            frame["home_score"].isna()
            | frame["away_score"].isna()
        )
    )


def update_frame(
    frame: pd.DataFrame,
    already_fetched: dict[str, ScoreFetchResult],
    stats: dict[str, int],
    source_name: str,
) -> tuple[pd.DataFrame, bool]:
    """Backfill one data frame without writing any non-final result."""
    if "game_id" not in frame.columns:
        print(f"Skipping {source_name}: missing game_id column.")
        stats["files_missing_game_id_column"] += 1
        return frame, False

    output = ensure_score_columns(frame)
    missing_mask = frame_missing_score_mask(output)
    changed = False

    for index, row in output.loc[missing_mask].iterrows():
        if stats["final_games_found"] >= TARGET_FINAL_GAMES:
            break

        stats["rows_scanned"] += 1
        game_id = normalize_game_id(row.get("game_id"))

        if not game_id:
            stats["invalid_game_id_rows"] += 1
            continue

        if game_id in already_fetched:
            stats["duplicate_rows_reused"] += 1
            result = already_fetched[game_id]
        else:
            stats["unique_games_requested"] += 1
            result = fetch_final_score(game_id)
            already_fetched[game_id] = result

            if result.status == "final_success":
                stats["final_games_found"] += 1
            elif result.status == "non_final":
                stats["non_final_games_skipped"] += 1
            elif result.status == "api_error":
                stats["api_errors"] += 1
            elif result.status == "missing_score":
                stats["final_games_missing_score"] += 1

            if REQUEST_SLEEP_SECONDS > 0:
                time.sleep(REQUEST_SLEEP_SECONDS)

        if result.status != "final_success":
            continue

        output.at[index, "home_win"] = result.home_win
        output.at[index, "home_score"] = result.home_score
        output.at[index, "away_score"] = result.away_score
        stats["rows_backfilled"] += 1
        changed = True

    return output, changed


def get_date_column(frame: pd.DataFrame) -> str | None:
    """Return the available date column name."""
    if "game_date" in frame.columns:
        return "game_date"

    if "date" in frame.columns:
        return "date"

    return None


def select_recent_csv_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select recent CSV rows while retaining original row indexes."""
    date_column = get_date_column(frame)

    if date_column is None:
        print(
            "Historical CSV has no date column; "
            f"using the last {MAX_UNDATED_CSV_ROWS} rows only."
        )
        return frame.tail(MAX_UNDATED_CSV_ROWS).copy()

    parsed_dates = pd.to_datetime(frame[date_column], errors="coerce")
    selected = frame.loc[parsed_dates >= MINIMUM_BACKFILL_DATE].copy()

    if selected.empty:
        print(
            "Historical CSV contains no rows on or after "
            f"{MINIMUM_BACKFILL_DATE.date()}."
        )
        return selected

    selected["_repair_date"] = parsed_dates.loc[selected.index]
    selected = selected.sort_values("_repair_date", ascending=False)
    selected = selected.drop(columns=["_repair_date"])

    return selected


def write_report(stats: dict[str, Any]) -> None:
    """Write a diagnostic report for GitHub Actions artifacts."""
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )


def process_history_csv(
    already_fetched: dict[str, ScoreFetchResult],
    stats: dict[str, Any],
) -> None:
    """Backfill recent rows in the main historical CSV file."""
    if not HISTORY_FILE.exists():
        stats["csv_file_exists"] = False
        return

    stats["csv_file_exists"] = True

    try:
        csv_frame = pd.read_csv(HISTORY_FILE)
    except Exception as exc:
        stats["files_read_failed"] += 1
        print(f"Unable to process {HISTORY_FILE}: {exc}")
        return

    recent_csv = select_recent_csv_rows(csv_frame)
    stats["csv_recent_rows_selected"] = int(len(recent_csv))

    if recent_csv.empty:
        return

    recent_csv, changed = update_frame(
        recent_csv,
        already_fetched,
        stats,
        str(HISTORY_FILE),
    )

    if not changed:
        return

    for column in ("home_win", "home_score", "away_score"):
        if column not in csv_frame.columns:
            csv_frame[column] = np.nan

        csv_frame.loc[recent_csv.index, column] = recent_csv[column]

    csv_frame.to_csv(HISTORY_FILE, index=False, encoding="utf-8")
    stats["files_updated"] += 1
    print(f"Updated {HISTORY_FILE}.")


def process_historical_parquet(
    already_fetched: dict[str, ScoreFetchResult],
    stats: dict[str, Any],
) -> None:
    """Backfill recent dated parquet files from newest date backwards."""
    if not HISTORICAL_DIR.exists():
        stats["historical_directory_exists"] = False
        return

    stats["historical_directory_exists"] = True

    parquet_paths = sorted(
        HISTORICAL_DIR.glob("*.parquet"),
        reverse=True,
    )

    stats["parquet_files_seen"] = int(len(parquet_paths))

    for parquet_path in parquet_paths:
        if stats["final_games_found"] >= TARGET_FINAL_GAMES:
            print(
                f"Target reached: {stats['final_games_found']} finalized games found."
            )
            break

        file_date = pd.to_datetime(parquet_path.stem, errors="coerce")

        if pd.notna(file_date) and file_date < MINIMUM_BACKFILL_DATE:
            stats["parquet_files_before_cutoff_skipped"] += 1
            continue

        stats["parquet_files_attempted"] += 1

        try:
            parquet_frame = pd.read_parquet(parquet_path)
        except Exception as exc:
            stats["files_read_failed"] += 1
            print(f"Unable to read {parquet_path}: {exc}")
            continue

        parquet_frame, changed = update_frame(
            parquet_frame,
            already_fetched,
            stats,
            str(parquet_path),
        )

        if changed:
            try:
                parquet_frame.to_parquet(parquet_path, index=False)
                stats["files_updated"] += 1
                stats["parquet_files_updated"] += 1
                print(f"Updated {parquet_path}.")
            except Exception as exc:
                stats["files_write_failed"] += 1
                print(f"Unable to write {parquet_path}: {exc}")


def backfill() -> None:
    """Backfill a bounded recent window of finalized MLB game results."""
    stats: dict[str, Any] = {
        "minimum_backfill_date": str(MINIMUM_BACKFILL_DATE.date()),
        "target_final_games": TARGET_FINAL_GAMES,
        "rows_scanned": 0,
        "rows_backfilled": 0,
        "unique_games_requested": 0,
        "final_games_found": 0,
        "non_final_games_skipped": 0,
        "api_errors": 0,
        "final_games_missing_score": 0,
        "duplicate_rows_reused": 0,
        "invalid_game_id_rows": 0,
        "files_missing_game_id_column": 0,
        "files_updated": 0,
        "files_read_failed": 0,
        "files_write_failed": 0,
        "parquet_files_seen": 0,
        "parquet_files_attempted": 0,
        "parquet_files_updated": 0,
        "parquet_files_before_cutoff_skipped": 0,
        "csv_recent_rows_selected": 0,
    }

    already_fetched: dict[str, ScoreFetchResult] = {}

    process_history_csv(already_fetched, stats)

    if stats["final_games_found"] < TARGET_FINAL_GAMES:
        process_historical_parquet(already_fetched, stats)

    write_report(stats)

    print("Final score backfill summary:")
    for name, value in stats.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    backfill()
