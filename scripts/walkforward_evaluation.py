from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

SNAPSHOTS_CSV = DATA_DIR / "prediction_snapshots.csv"
FINALIZED_CSV = DATA_DIR / "finalized_games.csv"

OUTPUT_JSON = REPORT_DIR / "walkforward_evaluation.json"
OUTPUT_CSV = REPORT_DIR / "walkforward_predictions.csv"

MIN_REQUIRED_OOS_PREDICTIONS = 300

CSV_COLUMNS = [
    "fold_id",
    "game_id",
    "game_date",
    "prediction_created_at",
    "model_prob",
    "market_prob",
    "outcome",
    "clv",
    "status",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    status = {"path": str(path), "exists": path.exists(), "rows": 0, "error": ""}
    if not path.exists():
        status["error"] = "file_missing"
        return None, status

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        status["error"] = str(exc)
        return None, status

    status["rows"] = int(len(frame))
    return frame, status


def _estimate_available_oos_predictions(snapshots: Optional[pd.DataFrame]) -> int:
    if snapshots is None or snapshots.empty:
        return 0

    required = {"game_id", "snapshot_created_at"}
    if not required.issubset(set(snapshots.columns)):
        return 0

    frame = snapshots.copy()
    if "home_win" in frame.columns:
        frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")
        frame = frame[frame["home_win"].isin([0, 1])]

    if frame.empty:
        return 0

    return int(frame["game_id"].astype(str).nunique())


def _write_empty_predictions_csv() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)


def build_walkforward_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = _utc_now()

    snapshots, snapshots_status = _safe_read_csv(SNAPSHOTS_CSV)
    _, finalized_status = _safe_read_csv(FINALIZED_CSV)

    input_files = {
        "prediction_snapshots": snapshots_status,
        "finalized_games": finalized_status,
    }

    total_oos = _estimate_available_oos_predictions(snapshots)
    walkforward_ready = total_oos >= MIN_REQUIRED_OOS_PREDICTIONS

    report = {
        "generated_at": generated_at,
        "status": "insufficient_samples" if not walkforward_ready else "partial",
        "input_files": input_files,
        "min_required_oos_predictions": MIN_REQUIRED_OOS_PREDICTIONS,
        "total_oos_predictions": int(total_oos),
        "walkforward_ready": bool(walkforward_ready),
        "fold_count": 0,
        "model_brier": None,
        "market_brier": None,
        "model_logloss": None,
        "market_logloss": None,
        "avg_clv": None,
        "positive_clv_rate": None,
        "model_beats_market": None,
        "recommendations": [
            "Walk-forward scaffold is active. Do not use walk-forward metrics for promotion until true rolling OOS folds are implemented and at least 300 OOS predictions are available."
        ],
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_empty_predictions_csv()
    return report


def main() -> None:
    report = build_walkforward_report()
    print(json.dumps({"status": report["status"], "output_path": str(OUTPUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
