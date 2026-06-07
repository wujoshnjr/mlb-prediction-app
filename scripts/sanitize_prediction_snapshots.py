from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


DATA_DIR = Path("data")
REPORT_DIR = Path("report")

SNAPSHOT_PATH = DATA_DIR / "prediction_snapshots.csv"
OUTPUT_REPORT = REPORT_DIR / "snapshot_sanitization_report.json"

POSTGAME_FIELDS = [
    "final_score",
    "home_final_score",
    "away_final_score",
    "home_score",
    "away_score",
    "settled_at",
    "actual_winner",
    "actual_result",
    "final_home_score",
    "final_away_score",
    "postgame_win_probability",
]

EARLY_RESULT_FIELDS = [
    "home_win",
    "outcome",
    "result",
    "winner",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_report(status: str, warnings: List[str], errors: List[str]) -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": _utc_now(),
        "status": status,
        "input_files": {
            "prediction_snapshots": {
                "path": str(SNAPSHOT_PATH),
                "exists": SNAPSHOT_PATH.exists(),
            }
        },
        "rows": 0,
        "pregame_rows": 0,
        "early_window_rows": 0,
        "cleared_cells": {},
        "errors": errors,
        "warnings": warnings,
        "recommendations": [
            "Pregame snapshots must not contain postgame or settled result fields."
        ],
    }
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not path.exists():
        return None, "file_missing"

    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return None, str(exc)


def _clear_columns(frame: pd.DataFrame, mask: pd.Series, columns: List[str]) -> Dict[str, int]:
    cleared: Dict[str, int] = {}

    for column in columns:
        if column not in frame.columns:
            continue

        affected = mask & frame[column].notna()
        count = int(affected.sum())

        if count > 0:
            frame.loc[affected, column] = pd.NA
            cleared[column] = count

    return cleared


def sanitize_snapshots() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    errors: List[str] = []

    frame, read_error = _safe_read_csv(SNAPSHOT_PATH)
    if frame is None:
        return _empty_report(
            status="partial",
            warnings=[f"prediction_snapshots.csv unavailable: {read_error}"],
            errors=[],
        )

    if frame.empty:
        return _empty_report(
            status="partial",
            warnings=["prediction_snapshots.csv is empty"],
            errors=[],
        )

    required = {"snapshot_created_at", "start_time"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return _empty_report(
            status="partial",
            warnings=[f"missing timestamp columns: {missing}"],
            errors=[],
        )

    snapshot_time = pd.to_datetime(frame["snapshot_created_at"], errors="coerce", utc=True)
    start_time = pd.to_datetime(frame["start_time"], errors="coerce", utc=True)

    valid_time = snapshot_time.notna() & start_time.notna()
    pregame_mask = valid_time & (snapshot_time <= start_time)

    # Result labels should not be present pregame or in the early post-start window.
    # Six hours is conservative for MLB games and avoids settled result leakage.
    early_window_mask = valid_time & (snapshot_time <= start_time + pd.Timedelta(hours=6))

    cleared_cells: Dict[str, int] = {}

    cleared_postgame = _clear_columns(frame, pregame_mask, POSTGAME_FIELDS)
    cleared_early = _clear_columns(frame, early_window_mask, EARLY_RESULT_FIELDS)

    cleared_cells.update(cleared_postgame)
    for key, value in cleared_early.items():
        cleared_cells[key] = cleared_cells.get(key, 0) + value

    if cleared_cells:
        frame.to_csv(SNAPSHOT_PATH, index=False)

    report = {
        "generated_at": _utc_now(),
        "status": "ok",
        "input_files": {
            "prediction_snapshots": {
                "path": str(SNAPSHOT_PATH),
                "exists": SNAPSHOT_PATH.exists(),
                "rows": int(len(frame)),
            }
        },
        "rows": int(len(frame)),
        "pregame_rows": int(pregame_mask.sum()),
        "early_window_rows": int(early_window_mask.sum()),
        "cleared_cells": cleared_cells,
        "errors": errors,
        "warnings": warnings,
        "recommendations": [
            "Keep settled results in finalized_games.csv, not in pregame prediction snapshots.",
            "Run this sanitizer before pytest no-leakage checks.",
        ],
    }

    OUTPUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = sanitize_snapshots()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
