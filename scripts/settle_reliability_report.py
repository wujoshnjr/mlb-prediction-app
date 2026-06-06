from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

PREDICTION_PATH = REPORT_DIR / "prediction.json"
SNAPSHOT_PATH = DATA_DIR / "prediction_snapshots.csv"
FINALIZED_PATH = DATA_DIR / "finalized_games.csv"
OUTPUT_PATH = REPORT_DIR / "settle_reliability_report.json"

STALE_PENDING_DAYS = 7


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Tuple[Optional[Any], Dict[str, Any]]:
    status = {"path": str(path), "exists": path.exists(), "rows": None, "error": ""}
    if not path.exists():
        status["error"] = "file_missing"
        return None, status

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return None, status

    return data, status


def _read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
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


def _extract_predictions(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        for key in ("predictions", "today_predictions", "games"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    return []


def _game_ids_from_predictions(predictions: List[Dict[str, Any]]) -> set[str]:
    game_ids = set()
    for item in predictions:
        value = item.get("game_id")
        if value is not None and str(value).strip():
            game_ids.add(str(value))
    return game_ids


def _game_ids_from_frame(frame: Optional[pd.DataFrame]) -> set[str]:
    if frame is None or frame.empty or "game_id" not in frame.columns:
        return set()
    return set(frame["game_id"].dropna().astype(str))


def _find_stale_pending(
    predictions: List[Dict[str, Any]],
    pending_game_ids: set[str],
) -> List[str]:
    now = pd.Timestamp.now(tz=timezone.utc)
    stale_ids: List[str] = []

    for item in predictions:
        game_id = str(item.get("game_id", "") or "")
        if not game_id or game_id not in pending_game_ids:
            continue

        game_date_value = item.get("game_date") or item.get("start_time")
        if not game_date_value:
            continue

        game_dt = pd.to_datetime(game_date_value, errors="coerce", utc=True)
        if pd.isna(game_dt):
            continue

        if game_dt < now - pd.Timedelta(days=STALE_PENDING_DAYS):
            stale_ids.append(game_id)

    return sorted(set(stale_ids))


def build_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    errors: List[str] = []
    warnings: List[str] = []
    missing_required_columns: List[str] = []

    prediction_data, prediction_status = _read_json(PREDICTION_PATH)
    snapshots, snapshot_status = _read_csv(SNAPSHOT_PATH)
    finalized, finalized_status = _read_csv(FINALIZED_PATH)

    input_files = {
        "prediction": prediction_status,
        "prediction_snapshots": snapshot_status,
        "finalized_games": finalized_status,
    }

    predictions = _extract_predictions(prediction_data)
    prediction_ids = _game_ids_from_predictions(predictions)
    snapshot_ids = _game_ids_from_frame(snapshots)
    finalized_ids = _game_ids_from_frame(finalized)

    if prediction_data is None:
        warnings.append("prediction.json missing or unreadable")
    if snapshots is None:
        warnings.append("prediction_snapshots.csv missing or unreadable")
    if finalized is None:
        warnings.append("finalized_games.csv missing or unreadable")

    if snapshots is not None and "game_id" not in snapshots.columns:
        missing_required_columns.append("prediction_snapshots.game_id")

    if finalized is not None and "game_id" not in finalized.columns:
        missing_required_columns.append("finalized_games.game_id")

    if prediction_data is not None and not predictions:
        warnings.append("prediction.json did not contain a prediction list")

    matched_settled = prediction_ids & finalized_ids
    unmatched_predictions = prediction_ids - finalized_ids
    pending_predictions = unmatched_predictions

    settle_rate = (
        len(matched_settled) / len(prediction_ids)
        if prediction_ids
        else 0.0
    )

    stale_pending_game_ids = _find_stale_pending(predictions, pending_predictions)

    if missing_required_columns:
        errors.append("required columns missing: " + ", ".join(missing_required_columns))

    if errors:
        status = "failed"
    elif not prediction_ids or finalized is None:
        status = "partial"
    else:
        status = "ok"

    recommendations: List[str] = []
    if stale_pending_game_ids:
        recommendations.append(
            "Investigate stale pending games that have not been matched to finalized results."
        )
    if not prediction_ids:
        recommendations.append("No current predictions available to assess settlement reliability.")
    if finalized is None:
        recommendations.append("finalized_games.csv is needed for settle-rate tracking.")

    report = {
        "generated_at": _utc_now(),
        "status": status,
        "input_files": input_files,
        "prediction_count": len(prediction_ids),
        "snapshot_game_count": len(snapshot_ids),
        "finalized_game_count": len(finalized_ids),
        "matched_settled_count": len(matched_settled),
        "unmatched_prediction_count": len(unmatched_predictions),
        "pending_prediction_count": len(pending_predictions),
        "settle_rate": round(settle_rate, 4),
        "stale_pending_count": len(stale_pending_game_ids),
        "stale_pending_game_ids": stale_pending_game_ids[:50],
        "missing_required_columns": missing_required_columns,
        "errors": errors,
        "warnings": warnings,
        "recommendations": recommendations,
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
