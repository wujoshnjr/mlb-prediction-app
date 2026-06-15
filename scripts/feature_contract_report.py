# scripts/feature_contract_report.py
"""Generate a feature-contract governance report.

Checks that scripts.feature_schema remains the single source of truth and that
the active training/serving files reference CORE_MODEL_FEATURES.
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
        DEFERRED_ZERO_MODEL_FEATURES,
        EXPECTED_FEATURES,
        MODEL_FEATURES,
        MODEL_FEATURE_VERSION,
        MODEL_FEATURE_SCHEMA_HASH,
        SHADOW_CANDIDATE_FEATURES,
        get_model_feature_schema_hash,
        validate_no_overlap,
    )
except ImportError:  # pragma: no cover
    from feature_schema import (  # type: ignore[no-redef]
        CORE_MODEL_FEATURES,
        DEFERRED_ZERO_MODEL_FEATURES,
        EXPECTED_FEATURES,
        MODEL_FEATURES,
        MODEL_FEATURE_VERSION,
        MODEL_FEATURE_SCHEMA_HASH,
        SHADOW_CANDIDATE_FEATURES,
        get_model_feature_schema_hash,
        validate_no_overlap,
    )


REPORT_PATH = Path("report/feature_contract_report.json")
TRAIN_ENSEMBLE_PATH = Path("train_ensemble.py")
PREDICTION_PATH = Path("prediction.py")
FEATURE_SCHEMA_PATH = Path("scripts/feature_schema.py")


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


def duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: set[str] = set()
    for value in values:
        if value in seen:
            output.add(value)
        seen.add(value)
    return sorted(output)


def file_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8")
    except Exception:
        return False


def base_report() -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "report_type": "feature_contract_report",
        "status": "ok",
        "feature_schema_path": str(FEATURE_SCHEMA_PATH),
        "train_ensemble_path": str(TRAIN_ENSEMBLE_PATH),
        "prediction_path": str(PREDICTION_PATH),
        "feature_schema_version": MODEL_FEATURE_VERSION,
        "feature_schema_hash": get_model_feature_schema_hash(),
        "declared_feature_schema_hash": MODEL_FEATURE_SCHEMA_HASH,
        "core_feature_count": len(CORE_MODEL_FEATURES),
        "expected_feature_count": len(EXPECTED_FEATURES),
        "deferred_zero_feature_count": len(DEFERRED_ZERO_MODEL_FEATURES),
        "shadow_candidate_feature_count": len(SHADOW_CANDIDATE_FEATURES),
        "checks": {},
        "errors": [],
        "warnings": [],
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }


def generate_report() -> dict[str, Any]:
    report = base_report()
    overlap_result = validate_no_overlap()
    duplicate_core_features = duplicates(list(CORE_MODEL_FEATURES))
    duplicate_expected_features = duplicates(list(EXPECTED_FEATURES))
    core_set = set(CORE_MODEL_FEATURES)
    deferred_set = set(DEFERRED_ZERO_MODEL_FEATURES)
    shadow_set = set(SHADOW_CANDIDATE_FEATURES)
    checks: dict[str, Any] = {
        "feature_schema_file_exists": FEATURE_SCHEMA_PATH.exists(),
        "core_no_duplicates": len(duplicate_core_features) == 0,
        "expected_no_duplicates": len(duplicate_expected_features) == 0,
        "model_features_alias_matches_core": list(MODEL_FEATURES) == list(CORE_MODEL_FEATURES),
        "deferred_no_overlap_with_core": len(core_set.intersection(deferred_set)) == 0,
        "shadow_no_overlap_with_core": len(core_set.intersection(shadow_set)) == 0,
        "validate_no_overlap_ok": bool(overlap_result.get("ok", False)),
        "computed_hash_matches_declared_hash": get_model_feature_schema_hash() == MODEL_FEATURE_SCHEMA_HASH,
        "train_ensemble_file_exists": TRAIN_ENSEMBLE_PATH.exists(),
        "prediction_file_exists": PREDICTION_PATH.exists(),
        "train_ensemble_references_core": file_contains(TRAIN_ENSEMBLE_PATH, "CORE_MODEL_FEATURES"),
        "prediction_references_core": file_contains(PREDICTION_PATH, "CORE_MODEL_FEATURES"),
        "train_ensemble_references_training_samples": file_contains(TRAIN_ENSEMBLE_PATH, "data/training_samples.csv"),
    }
    report["checks"] = checks
    report["schema_validation"] = overlap_result
    report["duplicate_core_features"] = duplicate_core_features
    report["duplicate_expected_features"] = duplicate_expected_features
    report["deferred_overlap_with_core"] = sorted(core_set.intersection(deferred_set))
    report["shadow_overlap_with_core"] = sorted(core_set.intersection(shadow_set))
    hard_fail_checks = [
        "feature_schema_file_exists",
        "core_no_duplicates",
        "expected_no_duplicates",
        "model_features_alias_matches_core",
        "deferred_no_overlap_with_core",
        "shadow_no_overlap_with_core",
        "validate_no_overlap_ok",
        "computed_hash_matches_declared_hash",
    ]
    warning_checks = [
        "train_ensemble_file_exists",
        "prediction_file_exists",
        "train_ensemble_references_core",
        "prediction_references_core",
        "train_ensemble_references_training_samples",
    ]
    failed_hard = [name for name in hard_fail_checks if not checks.get(name)]
    failed_warning = [name for name in warning_checks if not checks.get(name)]
    if failed_hard:
        report["status"] = "error"
        report["errors"].extend(failed_hard)
    elif failed_warning:
        report["status"] = "warning"
        report["warnings"].extend(failed_warning)
    else:
        report["status"] = "ok"
    write_json_report(report)
    return report


if __name__ == "__main__":
    generate_report()
