# scripts/feature_missingness_report.py
"""Generate feature missingness and zero-rate diagnostics.

This report is diagnostic only. It does not promote tracking-only or shadow
features into the active model.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.feature_schema import (
        CORE_MODEL_FEATURES,
        EXPECTED_FEATURES,
        FEATURE_METADATA,
        SHADOW_CANDIDATE_FEATURES,
        TRACKING_ONLY_FEATURES,
        MODEL_FEATURE_VERSION,
        get_model_feature_schema_hash,
    )
except ImportError:  # pragma: no cover
    from feature_schema import (  # type: ignore[no-redef]
        CORE_MODEL_FEATURES,
        EXPECTED_FEATURES,
        FEATURE_METADATA,
        SHADOW_CANDIDATE_FEATURES,
        TRACKING_ONLY_FEATURES,
        MODEL_FEATURE_VERSION,
        get_model_feature_schema_hash,
    )


DATA_PATH = Path("data/training_samples.csv")
REPORT_PATH = Path("report/feature_missingness_report.json")


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


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def base_report() -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "report_type": "feature_missingness_report",
        "status": "skipped",
        "input_path": str(DATA_PATH),
        "feature_schema_version": MODEL_FEATURE_VERSION,
        "feature_schema_hash": get_model_feature_schema_hash(),
        "sample_count": 0,
        "core_feature_count": len(CORE_MODEL_FEATURES),
        "expected_feature_count": len(EXPECTED_FEATURES),
        "features": [],
        "summary": {
            "missing_core_features": [],
            "features_with_inf": [],
            "features_all_missing": [],
            "features_high_missing_rate": [],
            "features_high_zero_rate": [],
        },
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


def recommended_action(
    *,
    feature: str,
    exists: bool,
    missing_rate: float,
    zero_rate: float | None,
    inf_count: int | None,
) -> str:
    if not exists:
        return "fix_required_core_missing" if feature in CORE_MODEL_FEATURES else "track_missing"
    if inf_count and inf_count > 0:
        return "investigate_inf_values"
    if missing_rate >= 1.0:
        return "fix_required_all_missing" if feature in CORE_MODEL_FEATURES else "tracking_only_all_missing"
    if feature in CORE_MODEL_FEATURES and missing_rate > 0.20:
        return "investigate_core_missingness"
    if feature in CORE_MODEL_FEATURES and zero_rate is not None and zero_rate > 0.95:
        return "investigate_core_zero_rate"
    if feature in TRACKING_ONLY_FEATURES:
        return "track_only_do_not_promote"
    return "ok"


def feature_entry(frame: pd.DataFrame, feature: str, total_rows: int) -> dict[str, Any]:
    exists = feature in frame.columns
    metadata = FEATURE_METADATA.get(feature, {})
    allow_in_main = feature in CORE_MODEL_FEATURES
    allow_in_shadow = feature in CORE_MODEL_FEATURES or feature in SHADOW_CANDIDATE_FEATURES
    if not exists:
        return {
            "feature": feature,
            "exists": False,
            "missing_count": int(total_rows),
            "missing_rate": 1.0 if total_rows else 0.0,
            "invalid_numeric_count": None,
            "invalid_numeric_rate": None,
            "zero_count": None,
            "zero_rate": None,
            "non_zero_rate": None,
            "inf_count": None,
            "unique_count": None,
            "allow_in_main_model": allow_in_main,
            "allow_in_shadow_model": allow_in_shadow,
            "feature_group": metadata.get("group", "unknown"),
            "leakage_risk": metadata.get("leakage_risk", "unknown"),
            "recommended_action": recommended_action(
                feature=feature,
                exists=False,
                missing_rate=1.0,
                zero_rate=None,
                inf_count=None,
            ),
        }
    raw = frame[feature]
    numeric = pd.to_numeric(raw, errors="coerce")
    raw_missing = raw.isna()
    invalid_numeric = numeric.isna() & ~raw_missing
    numeric_array = numeric.to_numpy(dtype=float, na_value=np.nan)
    finite_mask = np.isfinite(numeric_array)
    inf_mask = np.isinf(numeric_array)
    missing_count = int(raw_missing.sum() + invalid_numeric.sum())
    missing_rate = float(missing_count / total_rows) if total_rows else 0.0
    invalid_numeric_count = int(invalid_numeric.sum())
    invalid_numeric_rate = float(invalid_numeric_count / total_rows) if total_rows else 0.0
    inf_count = int(inf_mask.sum())
    zero_mask = finite_mask & (numeric_array == 0.0)
    zero_count = int(np.sum(zero_mask))
    zero_rate = float(zero_count / total_rows) if total_rows else 0.0
    non_zero_rate = float(np.sum(finite_mask & (numeric_array != 0.0)) / total_rows) if total_rows else 0.0
    unique_count = int(numeric.dropna().nunique())
    return {
        "feature": feature,
        "exists": True,
        "missing_count": missing_count,
        "missing_rate": missing_rate,
        "invalid_numeric_count": invalid_numeric_count,
        "invalid_numeric_rate": invalid_numeric_rate,
        "zero_count": zero_count,
        "zero_rate": zero_rate,
        "non_zero_rate": non_zero_rate,
        "inf_count": inf_count,
        "unique_count": unique_count,
        "allow_in_main_model": allow_in_main,
        "allow_in_shadow_model": allow_in_shadow,
        "feature_group": metadata.get("group", "unknown"),
        "leakage_risk": metadata.get("leakage_risk", "unknown"),
        "recommended_action": recommended_action(
            feature=feature,
            exists=True,
            missing_rate=missing_rate,
            zero_rate=zero_rate,
            inf_count=inf_count,
        ),
    }


def generate_report() -> dict[str, Any]:
    report = base_report()
    if not DATA_PATH.exists():
        return skip_report(report, "training_samples_missing")
    try:
        frame = pd.read_csv(DATA_PATH)
    except Exception as exc:
        return error_report(report, f"unable_to_read_training_samples:{exc}")
    total_rows = int(len(frame))
    report["sample_count"] = total_rows
    if frame.empty:
        return skip_report(report, "training_samples_empty")
    all_features = unique_preserve_order(
        list(EXPECTED_FEATURES) + list(CORE_MODEL_FEATURES) + list(SHADOW_CANDIDATE_FEATURES)
    )
    entries = [feature_entry(frame, feature, total_rows) for feature in all_features]
    report["features"] = entries
    missing_core_features = [
        entry["feature"] for entry in entries if entry["allow_in_main_model"] and not entry["exists"]
    ]
    features_with_inf = [
        entry["feature"] for entry in entries if entry["inf_count"] is not None and int(entry["inf_count"]) > 0
    ]
    features_all_missing = [
        entry["feature"] for entry in entries if float(entry["missing_rate"] or 0.0) >= 1.0
    ]
    features_high_missing_rate = [
        entry["feature"] for entry in entries if entry["exists"] and float(entry["missing_rate"] or 0.0) > 0.20
    ]
    features_high_zero_rate = [
        entry["feature"] for entry in entries if entry["exists"] and float(entry["zero_rate"] or 0.0) > 0.95
    ]
    report["summary"] = {
        "missing_core_features": missing_core_features,
        "features_with_inf": features_with_inf,
        "features_all_missing": features_all_missing,
        "features_high_missing_rate": features_high_missing_rate,
        "features_high_zero_rate": features_high_zero_rate,
    }
    if missing_core_features:
        report["status"] = "error"
        report["errors"].append("one_or_more_core_model_features_missing")
    elif features_with_inf:
        report["status"] = "warning"
        report["warnings"].append("one_or_more_features_contain_inf")
    elif features_high_missing_rate or features_high_zero_rate:
        report["status"] = "warning"
        report["warnings"].append("one_or_more_features_need_review")
    else:
        report["status"] = "ok"
    write_json_report(report)
    return report


if __name__ == "__main__":
    generate_report()
