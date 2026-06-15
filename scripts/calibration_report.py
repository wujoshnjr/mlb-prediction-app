# scripts/calibration_report.py
"""Generate a probability calibration report from OOS predictions.

Input:
    data/oos_predictions_with_labels.csv

Output:
    report/calibration_report.json
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OOS_PATH = Path("data/oos_predictions_with_labels.csv")
REPORT_PATH = Path("report/calibration_report.json")
N_BINS = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): clean_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json_value(child) for child in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        if hasattr(value, "item"):
            return clean_json_value(value.item())
    except Exception:
        pass
    return str(value)


def write_json_report(report: dict[str, Any], path: Path | None = None) -> None:
    output_path = path if path is not None else REPORT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            clean_json_value(report),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def base_report() -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "report_type": "calibration_report",
        "status": "skipped",
        "input_path": str(OOS_PATH),
        "sample_count": 0,
        "valid_sample_count": 0,
        "dropped_invalid_rows": 0,
        "probability_column": "",
        "target_column": "",
        "brier": None,
        "ece": None,
        "mce": None,
        "n_bins": N_BINS,
        "reliability_table": [],
        "warnings": [],
        "errors": [],
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }


def skip_report(report: dict[str, Any], reason: str) -> dict[str, Any]:
    report["status"] = "skipped"
    report["errors"].append(reason)
    write_json_report(report)
    return report


def error_report(report: dict[str, Any], reason: str) -> dict[str, Any]:
    report["status"] = "error"
    report["errors"].append(reason)
    write_json_report(report)
    return report


def first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = N_BINS,
) -> tuple[list[dict[str, Any]], float, float]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    table: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    sample_count = len(y_true)

    for index in range(n_bins):
        start = float(bin_edges[index])
        end = float(bin_edges[index + 1])
        if index == n_bins - 1:
            mask = (y_prob >= start) & (y_prob <= end)
        else:
            mask = (y_prob >= start) & (y_prob < end)
        count = int(np.sum(mask))
        if count == 0:
            table.append(
                {
                    "bin_start": round(start, 6),
                    "bin_end": round(end, 6),
                    "count": 0,
                    "avg_predicted_probability": None,
                    "observed_win_rate": None,
                    "calibration_gap": None,
                }
            )
            continue
        avg_pred = float(np.mean(y_prob[mask]))
        observed = float(np.mean(y_true[mask]))
        gap = float(abs(avg_pred - observed))
        ece += (count / sample_count) * gap
        mce = max(mce, gap)
        table.append(
            {
                "bin_start": round(start, 6),
                "bin_end": round(end, 6),
                "count": count,
                "avg_predicted_probability": avg_pred,
                "observed_win_rate": observed,
                "calibration_gap": gap,
            }
        )
    return table, float(ece), float(mce)


def generate_report() -> dict[str, Any]:
    report = base_report()
    if not OOS_PATH.exists():
        return skip_report(report, "oos_predictions_missing")
    try:
        frame = pd.read_csv(OOS_PATH)
    except Exception as exc:
        return error_report(report, f"unable_to_read_oos_predictions:{exc}")
    report["sample_count"] = int(len(frame))
    if frame.empty:
        return skip_report(report, "oos_predictions_empty")
    target_column = first_existing_column(frame, ["y_true", "home_win", "label"])
    probability_column = first_existing_column(
        frame,
        ["y_prob", "final_prob_home", "predicted_home_win_pct", "displayed_home_win_pct"],
    )
    if target_column is None or probability_column is None:
        return skip_report(report, "missing_target_or_probability_column")
    report["target_column"] = target_column
    report["probability_column"] = probability_column
    y_true = pd.to_numeric(frame[target_column], errors="coerce")
    y_prob = pd.to_numeric(frame[probability_column], errors="coerce")
    valid = (
        y_true.isin([0, 1])
        & y_prob.notna()
        & np.isfinite(y_prob.to_numpy(dtype=float, na_value=np.nan))
        & (y_prob >= 0.0)
        & (y_prob <= 1.0)
    )
    clean = frame.loc[valid].copy()
    report["valid_sample_count"] = int(len(clean))
    report["dropped_invalid_rows"] = int(len(frame) - len(clean))
    if clean.empty:
        return skip_report(report, "no_valid_calibration_rows")
    y_true_array = pd.to_numeric(clean[target_column], errors="coerce").to_numpy(dtype=int)
    y_prob_array = pd.to_numeric(clean[probability_column], errors="coerce").to_numpy(dtype=float)
    table, ece, mce = calibration_bins(y_true_array, y_prob_array, n_bins=N_BINS)
    report["status"] = "ok"
    if report["dropped_invalid_rows"]:
        report["status"] = "warning"
        report["warnings"].append("some_invalid_rows_were_dropped")
    report["brier"] = float(np.mean((y_prob_array - y_true_array) ** 2))
    report["ece"] = ece
    report["mce"] = mce
    report["reliability_table"] = table
    write_json_report(report)
    return report


if __name__ == "__main__":
    generate_report()
