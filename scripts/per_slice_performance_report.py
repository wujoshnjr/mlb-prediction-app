# scripts/per_slice_performance_report.py
"""Generate per-slice OOS performance diagnostics."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OOS_PATH = Path("data/oos_predictions_with_labels.csv")
REPORT_PATH = Path("report/per_slice_performance_report.json")
MIN_SLICE_SAMPLE_COUNT = 10

TARGET_CANDIDATES = ["y_true", "home_win", "label"]
PROBABILITY_CANDIDATES = [
    "y_prob",
    "final_prob_home",
    "predicted_home_win_pct",
    "displayed_home_win_pct",
]
SLICE_COLUMNS = [
    "month",
    "home_team",
    "away_team",
    "selected_side",
    "edge_bucket",
    "odds_quality_status",
    "model_source",
]


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
        "report_type": "per_slice_performance_report",
        "status": "skipped",
        "input_path": str(OOS_PATH),
        "sample_count": 0,
        "valid_sample_count": 0,
        "dropped_invalid_rows": 0,
        "target_column": "",
        "probability_column": "",
        "min_slice_sample_count": MIN_SLICE_SAMPLE_COUNT,
        "slices": {},
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


def prepare_frame(frame: pd.DataFrame, target_column: str, probability_column: str) -> pd.DataFrame:
    work = frame.copy()
    work["_target"] = pd.to_numeric(work[target_column], errors="coerce")
    work["_prob"] = pd.to_numeric(work[probability_column], errors="coerce")
    valid = (
        work["_target"].isin([0, 1])
        & work["_prob"].notna()
        & np.isfinite(work["_prob"].to_numpy(dtype=float, na_value=np.nan))
        & (work["_prob"] >= 0.0)
        & (work["_prob"] <= 1.0)
    )
    work = work.loc[valid].copy()
    work["_target"] = work["_target"].astype(int)
    work["_prob"] = work["_prob"].astype(float)
    work["_pred"] = (work["_prob"] >= 0.5).astype(int)
    if "game_date" in work.columns:
        parsed_dates = pd.to_datetime(work["game_date"], errors="coerce")
        work["month"] = parsed_dates.dt.month.astype("Int64").astype(str)
        work.loc[parsed_dates.isna(), "month"] = "unknown"
    return work


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    count = int(len(group))
    return {
        "count": count,
        "accuracy": float(np.mean(group["_target"] == group["_pred"])),
        "avg_prob": float(np.mean(group["_prob"])),
        "observed_rate": float(np.mean(group["_target"])),
        "brier": float(np.mean((group["_prob"] - group["_target"]) ** 2)),
        "low_sample": count < MIN_SLICE_SAMPLE_COUNT,
    }


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
    target_column = first_existing_column(frame, TARGET_CANDIDATES)
    probability_column = first_existing_column(frame, PROBABILITY_CANDIDATES)
    if target_column is None or probability_column is None:
        return skip_report(report, "missing_target_or_probability_column")
    report["target_column"] = target_column
    report["probability_column"] = probability_column
    clean = prepare_frame(frame, target_column, probability_column)
    report["valid_sample_count"] = int(len(clean))
    report["dropped_invalid_rows"] = int(len(frame) - len(clean))
    if clean.empty:
        return skip_report(report, "no_valid_slice_rows")
    slices: dict[str, Any] = {}
    for column in SLICE_COLUMNS:
        if column not in clean.columns:
            slices[column] = {"status": "skipped", "reason": "column_missing", "entries": []}
            continue
        entries: list[dict[str, Any]] = []
        for value, group in clean.groupby(column, dropna=False):
            summary = summarize_group(group)
            summary["value"] = "missing" if pd.isna(value) else str(value)
            entries.append(summary)
        entries.sort(key=lambda item: (-int(item["count"]), str(item["value"])))
        slices[column] = {"status": "ok" if entries else "skipped", "entries": entries}
    report["slices"] = slices
    report["status"] = "ok"
    if report["dropped_invalid_rows"]:
        report["status"] = "warning"
        report["warnings"].append("some_invalid_rows_were_dropped")
    if any(item.get("status") == "skipped" for item in slices.values()):
        report["warnings"].append("one_or_more_slice_columns_missing")
    write_json_report(report)
    return report


if __name__ == "__main__":
    generate_report()
