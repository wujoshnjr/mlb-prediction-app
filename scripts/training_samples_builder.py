#!/usr/bin/env python3
"""Build canonical clean training samples.

This script is part of the personal MLB Intelligence Engine evidence chain.

Source of truth:
- prediction_snapshots.csv provides pregame features only.
- finalized_snapshot_outcomes.csv provides trusted outcomes only.

The output data/training_samples.csv is the canonical source for model training.
No postgame leakage columns from prediction snapshots may survive into the output.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    from config import PIPELINE_VERSION, SNAPSHOT_POLICY
except Exception:
    PIPELINE_VERSION = "baseline_v2_clean"
    SNAPSHOT_POLICY = "first_seen_pregame"


SNAPSHOT_PATH = Path("data/prediction_snapshots.csv")
FINALIZED_SNAPSHOT_OUTCOMES_PATH = Path("data/finalized_snapshot_outcomes.csv")
TRAINING_SAMPLES_PATH = Path("data/training_samples.csv")
REPORT_PATH = Path("report/training_samples_report.json")

CORE_OUTPUT_COLUMNS = [
    "snapshot_id",
    "game_id",
    "game_date",
    "snapshot_created_at",
    "start_time",
    "home_team",
    "away_team",
    "home_win",
    "pipeline_version",
    "snapshot_policy",
    "market_no_vig_home_prob",
    "premarket_model_home_prob",
    "closing_market_home_prob",
    "edge_at_prediction_time",
    "odds_source",
    "data_quality_grade",
]

LEAKAGE_COLUMNS = {
    "home_win",
    "home_score",
    "away_score",
    "final_score",
    "home_final_score",
    "away_final_score",
    "final_home_score",
    "final_away_score",
    "settled_at",
    "actual_winner",
    "actual_result",
    "postgame_win_probability",
    "winning_pitcher",
    "losing_pitcher",
    "save_pitcher",
    "closing_home_odds",
    "closing_away_odds",
    "closing_spread_line",
    "closing_total_line",
    "clv_home_moneyline",
}

INTERNAL_COLUMNS = {
    "_snapshot_created_at_ts",
    "_start_time_ts",
    "_snapshot_valid_bool",
    "_normalized_game_id",
    "_outcome_home_win",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value) and not isinstance(value, (str, bool, dict, list, tuple)):
            return None
    except Exception:
        pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "rows": 0,
        "error": "",
    }

    if not path.exists():
        status["error"] = "file_missing"
        return pd.DataFrame(), status

    try:
        frame = pd.read_csv(path, dtype=str)
    except Exception as exc:
        status["error"] = str(exc)
        return pd.DataFrame(), status

    status["rows"] = int(len(frame))
    return frame, status


def _normalize_game_id(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return ""

    try:
        parsed = float(text)
        if math.isfinite(parsed) and parsed.is_integer():
            return str(int(parsed))
    except Exception:
        pass

    return text


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(True)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "valid", "ok"})


def _empty_training_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CORE_OUTPUT_COLUMNS)


def _write_outputs(
    training_samples: pd.DataFrame,
    output_path: Path,
    report_path: Path,
    report: Dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if training_samples.empty:
        training_samples = _empty_training_frame()

    training_samples.to_csv(output_path, index=False)
    _write_json(report_path, report)


def _derive_status(report: Dict[str, Any]) -> str:
    if report.get("errors"):
        return "partial"
    if int(report.get("clean_training_rows", 0) or 0) == 0:
        return "partial"
    return "ok"


def _prepare_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty or "game_id" not in outcomes.columns:
        return pd.DataFrame(columns=["game_id", "_outcome_home_win"])

    result = outcomes.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()

    if "home_win" not in result.columns:
        if {"home_score", "away_score"}.issubset(result.columns):
            home_score = pd.to_numeric(result["home_score"], errors="coerce")
            away_score = pd.to_numeric(result["away_score"], errors="coerce")
            result["home_win"] = (home_score > away_score).astype("Int64")
        else:
            result["home_win"] = pd.NA

    result["home_win"] = pd.to_numeric(result["home_win"], errors="coerce")
    result = result[result["home_win"].isin([0, 1])].copy()

    if result.empty:
        return pd.DataFrame(columns=["game_id", "_outcome_home_win"])

    result = result.drop_duplicates("game_id", keep="last").copy()
    result["_outcome_home_win"] = result["home_win"].astype(int)

    return result[["game_id", "_outcome_home_win"]].copy()


def _prepare_snapshots(
    snapshots: pd.DataFrame,
    *,
    pipeline_version: str,
    report: Dict[str, Any],
) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()

    required_columns = ["game_id", "snapshot_created_at", "start_time", "pipeline_version"]
    missing = [column for column in required_columns if column not in snapshots.columns]
    if missing:
        report["errors"].append(
            "prediction_snapshots missing required columns: " + ", ".join(missing)
        )
        return pd.DataFrame()

    result = snapshots.copy()
    result["_normalized_game_id"] = result["game_id"].apply(_normalize_game_id)

    before = len(result)
    result = result[result["_normalized_game_id"] != ""].copy()
    report["dropped_empty_game_id_rows"] = int(before - len(result))

    result["game_id"] = result["_normalized_game_id"]

    result["_snapshot_created_at_ts"] = pd.to_datetime(
        result["snapshot_created_at"],
        errors="coerce",
        utc=True,
    )
    result["_start_time_ts"] = pd.to_datetime(
        result["start_time"],
        errors="coerce",
        utc=True,
    )

    before = len(result)
    result = result[
        result["_snapshot_created_at_ts"].notna()
        & result["_start_time_ts"].notna()
        & (result["_snapshot_created_at_ts"] < result["_start_time_ts"])
    ].copy()
    report["dropped_post_start_rows"] = int(before - len(result))

    if "snapshot_valid" in result.columns:
        before = len(result)
        result["_snapshot_valid_bool"] = _to_bool_series(result["snapshot_valid"])
        result = result[result["_snapshot_valid_bool"]].copy()
        report["dropped_invalid_snapshot_rows"] = int(before - len(result))
    else:
        report["warnings"].append("snapshot_valid column missing; treating snapshots as valid.")
        result["_snapshot_valid_bool"] = True
        report["dropped_invalid_snapshot_rows"] = 0

    before = len(result)
    result = result[result["pipeline_version"].astype(str) == str(pipeline_version)].copy()
    report["dropped_pipeline_mismatch_rows"] = int(before - len(result))

    if result.empty:
        return pd.DataFrame()

    result = result.sort_values("_snapshot_created_at_ts")
    result = result.drop_duplicates("game_id", keep="first").copy()

    return result.reset_index(drop=True)


def _select_output_columns(frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    leakage_removed = sorted(
        column for column in frame.columns if column in LEAKAGE_COLUMNS
    )

    available_core = [
        column for column in CORE_OUTPUT_COLUMNS if column in frame.columns
    ]

    extra_columns = []
    for column in frame.columns:
        if column in available_core:
            continue
        if column in LEAKAGE_COLUMNS:
            continue
        if column in INTERNAL_COLUMNS:
            continue
        if column.endswith("_x") or column.endswith("_y"):
            # Suffixes usually indicate an unintended duplicate merge column.
            # Do not allow them into model training silently.
            continue
        extra_columns.append(column)

    return available_core + extra_columns, leakage_removed


def build_training_samples(
    snapshot_path: Path = SNAPSHOT_PATH,
    finalized_snapshot_outcomes_path: Path = FINALIZED_SNAPSHOT_OUTCOMES_PATH,
    output_path: Path = TRAINING_SAMPLES_PATH,
    report_path: Path = REPORT_PATH,
    pipeline_version: str = PIPELINE_VERSION,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "status": "partial",
        "pipeline_version": pipeline_version,
        "snapshot_policy": SNAPSHOT_POLICY,
        "raw_snapshot_rows": 0,
        "finalized_outcome_rows": 0,
        "linked_rows": 0,
        "clean_training_rows": 0,
        "dropped_empty_game_id_rows": 0,
        "dropped_post_start_rows": 0,
        "dropped_invalid_snapshot_rows": 0,
        "dropped_pipeline_mismatch_rows": 0,
        "dropped_missing_outcome_rows": 0,
        "leakage_columns_removed": [],
        "output_path": str(output_path),
        "input_files": {},
        "errors": [],
        "warnings": [],
        "recommendations": [],
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }

    snapshots_raw, snapshot_status = _read_csv(snapshot_path)
    outcomes_raw, outcome_status = _read_csv(finalized_snapshot_outcomes_path)

    report["input_files"] = {
        "prediction_snapshots": snapshot_status,
        "finalized_snapshot_outcomes": outcome_status,
    }
    report["raw_snapshot_rows"] = int(len(snapshots_raw))
    report["finalized_outcome_rows"] = int(len(outcomes_raw))

    if snapshot_status["error"]:
        report["warnings"].append(f"prediction_snapshots unavailable: {snapshot_status['error']}")

    if outcome_status["error"]:
        report["warnings"].append(
            f"finalized_snapshot_outcomes unavailable: {outcome_status['error']}"
        )

    snapshots = _prepare_snapshots(
        snapshots_raw,
        pipeline_version=pipeline_version,
        report=report,
    )
    outcomes = _prepare_outcomes(outcomes_raw)

    if snapshots.empty:
        report["warnings"].append("No valid pregame snapshots available for training sample construction.")
        report["status"] = _derive_status(report)
        _write_outputs(_empty_training_frame(), output_path, report_path, report)
        return report

    if outcomes.empty:
        report["warnings"].append("No valid finalized snapshot outcomes available.")
        report["dropped_missing_outcome_rows"] = int(len(snapshots))
        report["status"] = _derive_status(report)
        _write_outputs(_empty_training_frame(), output_path, report_path, report)
        return report

    merged = snapshots.merge(outcomes, on="game_id", how="inner")
    report["linked_rows"] = int(len(merged))
    report["dropped_missing_outcome_rows"] = int(len(snapshots) - len(merged))

    if merged.empty:
        report["warnings"].append("No valid snapshots linked to finalized outcomes.")
        report["status"] = _derive_status(report)
        _write_outputs(_empty_training_frame(), output_path, report_path, report)
        return report

    merged["home_win"] = pd.to_numeric(merged["_outcome_home_win"], errors="coerce")
    merged = merged[merged["home_win"].isin([0, 1])].copy()
    merged["home_win"] = merged["home_win"].astype(int)

    if "snapshot_created_at" in merged.columns:
        merged["snapshot_created_at"] = pd.to_datetime(
            merged["snapshot_created_at"],
            errors="coerce",
            utc=True,
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if "start_time" in merged.columns:
        merged["start_time"] = pd.to_datetime(
            merged["start_time"],
            errors="coerce",
            utc=True,
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    output_columns, leakage_removed = _select_output_columns(merged)
    report["leakage_columns_removed"] = leakage_removed

    if "home_win" not in output_columns:
        insertion_index = output_columns.index("away_team") + 1 if "away_team" in output_columns else len(output_columns)
        output_columns.insert(insertion_index, "home_win")

    training_samples = merged[output_columns].copy()
    training_samples = training_samples.drop_duplicates("game_id", keep="first").copy()

    report["clean_training_rows"] = int(len(training_samples))

    if training_samples.empty:
        report["warnings"].append("Training samples output is empty after final filtering.")

    report["status"] = _derive_status(report)
    _write_outputs(training_samples, output_path, report_path, report)
    return report


def main() -> None:
    report = build_training_samples()
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
