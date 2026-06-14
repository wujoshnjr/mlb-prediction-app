from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_PATH = Path("report/feature_contract_report.json")
TRAIN_ENSEMBLE_PATH = Path("train_ensemble.py")
PREDICTION_PATH = Path("prediction.py")
MODEL_PATH = Path("model.py")

REPORT_TYPE = "feature_contract_v1"


try:
    from scripts.feature_schema import (
        CORE_MODEL_FEATURES,
        DEFERRED_ZERO_MODEL_FEATURES,
        SHADOW_CANDIDATE_FEATURES,
        MODEL_FEATURE_VERSION,
        get_model_feature_schema_hash,
        validate_no_overlap,
    )

    FEATURE_SCHEMA_IMPORT_ERROR = ""
except Exception as exc:
    CORE_MODEL_FEATURES = []
    DEFERRED_ZERO_MODEL_FEATURES = []
    SHADOW_CANDIDATE_FEATURES = []
    MODEL_FEATURE_VERSION = "unknown"
    FEATURE_SCHEMA_IMPORT_ERROR = str(exc)

    def get_model_feature_schema_hash() -> str:
        return ""

    def validate_no_overlap() -> dict[str, Any]:
        return {
            "ok": False,
            "errors": [FEATURE_SCHEMA_IMPORT_ERROR],
            "warnings": [],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    try:
        if hasattr(value, "item"):
            return _json_safe(value.item())
    except Exception:
        pass

    return str(value)


def safe_json_dump(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_data = _json_safe(data)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            safe_data,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def _find_duplicate_items(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)

    return sorted(duplicates)


def _read_text_safe(path: Path) -> tuple[str, dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "error": "",
    }

    if not path.exists():
        status["error"] = "file not found"
        return "", status

    try:
        return path.read_text(encoding="utf-8", errors="ignore"), status
    except Exception as exc:
        status["error"] = str(exc)
        return "", status


def _file_contains(path: Path, needles: list[str]) -> tuple[bool, dict[str, Any]]:
    content, status = _read_text_safe(path)

    if status["error"]:
        return False, status

    return any(needle in content for needle in needles), status


def _generate_static_source_status() -> dict[str, Any]:
    train_uses_schema, train_status = _file_contains(
        TRAIN_ENSEMBLE_PATH,
        [
            "CORE_MODEL_FEATURES",
            "get_core_model_features",
            "scripts.feature_schema",
        ],
    )

    train_uses_canonical_source, train_source_status = _file_contains(
        TRAIN_ENSEMBLE_PATH,
        [
            "data/training_samples.csv",
            "TRAINING_SAMPLES_FILE",
            "TRAINING_SOURCE",
        ],
    )

    serving_uses_schema, prediction_status = _file_contains(
        PREDICTION_PATH,
        [
            "CORE_MODEL_FEATURES",
            "get_core_model_features",
            "scripts.feature_schema",
        ],
    )

    model_uses_schema, model_status = _file_contains(
        MODEL_PATH,
        [
            "CORE_MODEL_FEATURES",
            "get_core_model_features",
            "scripts.feature_schema",
        ],
    )

    return {
        "train_ensemble": {
            "path": str(TRAIN_ENSEMBLE_PATH),
            "uses_feature_schema": bool(train_uses_schema),
            "uses_canonical_training_source": bool(train_uses_canonical_source),
            "status": train_status,
            "training_source_status": train_source_status,
        },
        "prediction": {
            "path": str(PREDICTION_PATH),
            "uses_feature_schema": bool(serving_uses_schema),
            "status": prediction_status,
        },
        "model": {
            "path": str(MODEL_PATH),
            "uses_feature_schema": bool(model_uses_schema),
            "status": model_status,
        },
    }


def generate_report() -> dict[str, Any]:
    core_features = list(CORE_MODEL_FEATURES)
    deferred_features = list(DEFERRED_ZERO_MODEL_FEATURES)
    shadow_features = list(SHADOW_CANDIDATE_FEATURES)

    duplicate_core_features = _find_duplicate_items(core_features)
    duplicate_deferred_features = _find_duplicate_items(deferred_features)
    duplicate_shadow_features = _find_duplicate_items(shadow_features)

    deferred_overlap_with_core = sorted(
        set(core_features).intersection(set(deferred_features))
    )
    shadow_overlap_with_core = sorted(
        set(core_features).intersection(set(shadow_features))
    )

    warnings: list[str] = []
    errors: list[str] = []

    if FEATURE_SCHEMA_IMPORT_ERROR:
        errors.append(f"feature_schema import failed: {FEATURE_SCHEMA_IMPORT_ERROR}")

    try:
        feature_schema_hash = get_model_feature_schema_hash()
    except Exception as exc:
        feature_schema_hash = ""
        errors.append(f"feature schema hash error: {exc}")

    try:
        overlap_status = validate_no_overlap()
        if isinstance(overlap_status, dict):
            for warning in overlap_status.get("warnings", []):
                warnings.append(str(warning))
            for error in overlap_status.get("errors", []):
                errors.append(str(error))
    except Exception as exc:
        warnings.append(f"validate_no_overlap unavailable: {exc}")

    source_status = _generate_static_source_status()

    train_uses_schema = bool(
        source_status["train_ensemble"]["uses_feature_schema"]
    )
    train_uses_canonical_source = bool(
        source_status["train_ensemble"]["uses_canonical_training_source"]
    )
    serving_uses_schema = bool(
        source_status["prediction"]["uses_feature_schema"]
    )

    missing_in_training: list[str] = []
    missing_in_serving: list[str] = []

    if not train_uses_schema:
        missing_in_training.append("train_ensemble.py does not reference CORE_MODEL_FEATURES")

    if not train_uses_canonical_source:
        missing_in_training.append(
            "train_ensemble.py does not clearly reference data/training_samples.csv"
        )

    if not serving_uses_schema:
        missing_in_serving.append("prediction.py does not reference CORE_MODEL_FEATURES")

    if duplicate_core_features:
        errors.append(f"duplicate core features: {duplicate_core_features}")

    if duplicate_deferred_features:
        errors.append(f"duplicate deferred zero features: {duplicate_deferred_features}")

    if duplicate_shadow_features:
        errors.append(f"duplicate shadow candidate features: {duplicate_shadow_features}")

    if deferred_overlap_with_core:
        errors.append(
            f"CORE_MODEL_FEATURES overlaps DEFERRED_ZERO_MODEL_FEATURES: {deferred_overlap_with_core}"
        )

    if shadow_overlap_with_core:
        errors.append(
            f"CORE_MODEL_FEATURES overlaps SHADOW_CANDIDATE_FEATURES: {shadow_overlap_with_core}"
        )

    if missing_in_training:
        errors.extend(missing_in_training)

    if missing_in_serving:
        errors.extend(missing_in_serving)

    train_serving_consistent = (
        train_uses_schema
        and train_uses_canonical_source
        and serving_uses_schema
        and not duplicate_core_features
        and not deferred_overlap_with_core
        and not shadow_overlap_with_core
    )

    if not core_features:
        warnings.append("CORE_MODEL_FEATURES is empty.")

    report: dict[str, Any] = {
        "generated_at": _utc_now(),
        "status": "ok",
        "report_type": REPORT_TYPE,
        "feature_schema_version": str(MODEL_FEATURE_VERSION),
        "feature_schema_hash": feature_schema_hash,
        "core_model_features": core_features,
        "core_feature_count": len(core_features),
        "deferred_zero_model_features": deferred_features,
        "shadow_candidate_features": shadow_features,
        "train_feature_source": (
            "scripts.feature_schema.CORE_MODEL_FEATURES"
            if train_uses_schema
            else "unknown"
        ),
        "serving_feature_source": (
            "scripts.feature_schema.CORE_MODEL_FEATURES"
            if serving_uses_schema
            else "unknown"
        ),
        "train_serving_consistent": bool(train_serving_consistent),
        "missing_in_training": missing_in_training,
        "missing_in_serving": missing_in_serving,
        "deferred_overlap_with_core": deferred_overlap_with_core,
        "shadow_overlap_with_core": shadow_overlap_with_core,
        "duplicate_core_features": duplicate_core_features,
        "duplicate_deferred_zero_features": duplicate_deferred_features,
        "duplicate_shadow_candidate_features": duplicate_shadow_features,
        "source_status": source_status,
        "warnings": warnings,
        "errors": errors,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }

    if errors:
        report["status"] = "failed"
    elif warnings:
        report["status"] = "warning"
    else:
        report["status"] = "ok"

    safe_json_dump(report, REPORT_PATH)
    return report


def main() -> int:
    report = generate_report()
    print(
        json.dumps(
            _json_safe(report),
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
