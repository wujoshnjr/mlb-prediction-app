from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

PREDICTION_JSON = REPORT_DIR / "prediction.json"
SNAPSHOTS_CSV = DATA_DIR / "prediction_snapshots.csv"
FINALIZED_CSV = DATA_DIR / "finalized_games.csv"

OUTPUT_JSON = REPORT_DIR / "calibration_report.json"

BINS = [
    (0.35, 0.40),
    (0.40, 0.45),
    (0.45, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    status = {"path": str(path), "exists": path.exists(), "rows": None, "error": ""}
    if not path.exists():
        status["error"] = "file_missing"
        return None, status

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return None, status

    if not isinstance(data, dict):
        status["error"] = "json_not_object"
        return None, status

    predictions = data.get("predictions") or data.get("today_predictions") or data.get("games") or []
    status["rows"] = len(predictions) if isinstance(predictions, list) else None
    return data, status


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


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        parsed = float(value)
        if not math.isfinite(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    parsed = _as_float(value)
    if parsed is None:
        return default
    return int(parsed)


def _probability(value: Any) -> Optional[float]:
    parsed = _as_float(value)
    if parsed is None:
        return None
    if parsed > 1.0 and parsed <= 100.0:
        parsed = parsed / 100.0
    if parsed < 0.0 or parsed > 1.0:
        return None
    return float(parsed)


def _normalise_game_id(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "game_id" in frame.columns:
        frame["game_id"] = frame["game_id"].astype(str)
    return frame


def _prepare_settled_predictions(
    snapshots: Optional[pd.DataFrame],
    finalized: Optional[pd.DataFrame],
) -> List[Tuple[float, int]]:
    if snapshots is None or snapshots.empty or "game_id" not in snapshots.columns:
        return []

    if finalized is None or finalized.empty or "game_id" not in finalized.columns:
        return []

    frame = _normalise_game_id(snapshots)
    final = _normalise_game_id(finalized)

    if "home_win" not in final.columns:
        if {"home_score", "away_score"}.issubset(final.columns):
            final["home_win"] = (
                pd.to_numeric(final["home_score"], errors="coerce")
                > pd.to_numeric(final["away_score"], errors="coerce")
            ).astype("Int64")
        else:
            return []

    final["home_win"] = pd.to_numeric(final["home_win"], errors="coerce")
    final = final[final["home_win"].isin([0, 1])].copy()
    if final.empty:
        return []

    leaked_snapshot_columns = [
        column
        for column in ["home_win", "home_score", "away_score"]
        if column in frame.columns
    ]
    if leaked_snapshot_columns:
        frame = frame.drop(columns=leaked_snapshot_columns)

    frame = frame.merge(
        final[["game_id", "home_win"]].drop_duplicates("game_id"),
        on="game_id",
        how="inner",
    )

    if frame.empty:
        return []

    if "snapshot_created_at" in frame.columns:
        frame["_snapshot_dt"] = pd.to_datetime(
            frame["snapshot_created_at"],
            errors="coerce",
            utc=True,
        )
        frame = frame.sort_values("_snapshot_dt").groupby("game_id", as_index=False).tail(1)

    result: List[Tuple[float, int]] = []

    probability_columns = [
        "displayed_home_win_pct",
        "predicted_home_win_pct",
        "premarket_model_home_prob",
        "model_prob",
        "home_win_probability",
    ]

    for _, row in frame.iterrows():
        prob = None
        for column in probability_columns:
            if column in row.index:
                prob = _probability(row.get(column))
                if prob is not None:
                    break

        outcome = _as_int(row.get("home_win"))
        if prob is None or outcome not in (0, 1):
            continue

        result.append((prob, int(outcome)))

    return result
    

def _empty_bins() -> List[Dict[str, Any]]:
    return [
        {
            "bin": f"{low:.2f}-{high:.2f}",
            "count": 0,
            "avg_predicted": None,
            "actual_win_rate": None,
            "calibration_error": None,
        }
        for low, high in BINS
    ]


def build_calibration_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = _utc_now()

    _, prediction_status = _safe_read_json(PREDICTION_JSON)
    snapshots, snapshots_status = _safe_read_csv(SNAPSHOTS_CSV)
    finalized, finalized_status = _safe_read_csv(FINALIZED_CSV)

    input_files = {
        "prediction": prediction_status,
        "prediction_snapshots": snapshots_status,
        "finalized_games": finalized_status,
    }

    settled = _prepare_settled_predictions(snapshots, finalized)
    total_count = len(settled)

    if total_count == 0:
        report = {
            "generated_at": generated_at,
            "status": "insufficient_samples",
            "input_files": input_files,
            "calibration_ready": False,
            "total_count": 0,
            "bins": _empty_bins(),
            "weighted_ece": None,
            "max_calibration_error": None,
            "min_recommended_samples": 500,
            "recommendations": [
                "No settled predictions with model probability and outcome were available."
            ],
        }
        OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        return report

    bin_outputs: List[Dict[str, Any]] = []
    weighted_errors: List[Tuple[int, float]] = []

    for low, high in BINS:
        values = [(prob, outcome) for prob, outcome in settled if low <= prob < high]
        if not values:
            bin_outputs.append(
                {
                    "bin": f"{low:.2f}-{high:.2f}",
                    "count": 0,
                    "avg_predicted": None,
                    "actual_win_rate": None,
                    "calibration_error": None,
                }
            )
            continue

        probs = np.asarray([item[0] for item in values], dtype=float)
        outcomes = np.asarray([item[1] for item in values], dtype=float)
        avg_predicted = float(probs.mean())
        actual_rate = float(outcomes.mean())
        error = abs(avg_predicted - actual_rate)

        weighted_errors.append((len(values), error))

        bin_outputs.append(
            {
                "bin": f"{low:.2f}-{high:.2f}",
                "count": int(len(values)),
                "avg_predicted": round(avg_predicted, 6),
                "actual_win_rate": round(actual_rate, 6),
                "calibration_error": round(float(error), 6),
            }
        )

    if weighted_errors:
        weighted_ece = sum(count * err for count, err in weighted_errors) / sum(
            count for count, _ in weighted_errors
        )
        max_error = max(err for _, err in weighted_errors)
    else:
        weighted_ece = None
        max_error = None

    calibration_ready = total_count >= 500

    report = {
        "generated_at": generated_at,
        "status": "ok" if total_count >= 10 else "insufficient_samples",
        "input_files": input_files,
        "calibration_ready": bool(calibration_ready),
        "total_count": int(total_count),
        "bins": bin_outputs,
        "weighted_ece": round(float(weighted_ece), 6) if weighted_ece is not None else None,
        "max_calibration_error": round(float(max_error), 6) if max_error is not None else None,
        "min_recommended_samples": 500,
        "recommendations": []
        if calibration_ready
        else ["Need at least 500 settled predictions before calibration should be used for promotion."],
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_calibration_report()
    print(json.dumps({"status": report["status"], "output_path": str(OUTPUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
