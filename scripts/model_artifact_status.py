#!/usr/bin/env python3
"""Build model artifact status.

This script must never crash simply because an artifact is missing, too small,
or unreadable. Those are expected states during evidence collection.

It writes:
- data/model_artifact_status.json
- report/model_artifact_status_report.json
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import joblib
except Exception:
    joblib = None

try:
    from config import MIN_CLEAN_TRAIN_SAMPLES, PIPELINE_VERSION
except Exception:
    MIN_CLEAN_TRAIN_SAMPLES = 300
    PIPELINE_VERSION = "baseline_v2_clean"

from scripts.feature_schema import (
    CORE_MODEL_FEATURES,
    get_model_feature_schema_hash,
)

CALIBRATOR_PATH = Path("data/calibrator.pkl")
MODEL_DIR_ARTIFACT_PATH = Path("data/models/baseline_v2_clean/model.joblib")
TRAINING_SAMPLES_PATH = Path("data/training_samples.csv")
TRAINING_STATUS_PATH = Path("data/training_status.json")
MODEL_ARTIFACT_STATUS_PATH = Path("data/model_artifact_status.json")
REPORT_PATH = Path("report/model_artifact_status_report.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
    except Exception:
        pass
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "error": "",
    }

    if not path.exists():
        status["error"] = "file_missing"
        return {}, status

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return {}, status

    if not isinstance(payload, dict):
        status["error"] = "json_not_object"
        return {}, status

    return payload, status


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return int(parsed)
    except Exception:
        return None


def _count_csv_data_rows(path: Path) -> Tuple[Optional[int], Dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "rows": None,
        "error": "",
    }

    if not path.exists():
        status["error"] = "file_missing"
        return None, status

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                next(reader)
            except StopIteration:
                status["rows"] = 0
                return 0, status

            row_count = sum(1 for _ in reader)

        status["rows"] = int(row_count)
        return int(row_count), status

    except Exception as exc:
        status["error"] = str(exc)
        return None, status


def _select_artifact_path(
    artifact_path: Path,
    model_dir_artifact_path: Path,
) -> Path:
    if model_dir_artifact_path.exists():
        return model_dir_artifact_path
    return artifact_path


def _load_artifact(path: Path) -> Tuple[Optional[Any], str]:
    if not path.exists():
        return None, "artifact_missing"

    if joblib is not None:
        try:
            return joblib.load(path), ""
        except Exception as exc:
            joblib_error = str(exc)
    else:
        joblib_error = "joblib_unavailable"

    try:
        with path.open("rb") as handle:
            return pickle.load(handle), ""
    except Exception as exc:
        return None, f"joblib_error={joblib_error}; pickle_error={exc}"


def _extract_metadata(artifact: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    if artifact is None:
        return metadata

    if isinstance(artifact, dict):
        nested = artifact.get("metadata")
        if isinstance(nested, dict):
            metadata.update(nested)

        for key in [
            "artifact_version",
            "version",
            "pipeline_version",
            "schema_version",
            "model_type",
            "training_source",
            "training_sample_count",
            "sample_count",
            "n_samples",
            "feature_schema_hash",
            "feature_schema_version",
            "core_model_features",
            "core_feature_count",
            "feature_names",
            "features",
        ]:
            if key in artifact and artifact.get(key) is not None:
                metadata[key] = artifact.get(key)

        return metadata

    nested = getattr(artifact, "metadata", None)
    if isinstance(nested, dict):
        metadata.update(nested)

    for key in [
        "artifact_version",
        "version",
        "pipeline_version",
        "schema_version",
        "model_type",
        "training_source",
        "training_sample_count",
        "sample_count",
        "n_samples",
        "feature_schema_hash",
        "feature_schema_version",
        "core_model_features",
        "core_feature_count",
        "feature_names",
        "features",
    ]:
        try:
            value = getattr(artifact, key, None)
        except Exception:
            value = None
        if value is not None:
            metadata[key] = value

    return metadata


def _extract_feature_names(metadata: Dict[str, Any]) -> list[str]:
    feature_names = metadata.get("feature_names")
    if feature_names is None:
        feature_names = metadata.get("features")

    if isinstance(feature_names, list):
        return [str(item) for item in feature_names]

    if isinstance(feature_names, tuple):
        return [str(item) for item in feature_names]

    return []


def _extract_training_sample_count(
    metadata: Dict[str, Any],
    training_status: Dict[str, Any],
) -> int:
    candidates = [
        metadata.get("training_sample_count"),
        metadata.get("sample_count"),
        metadata.get("n_samples"),
        training_status.get("artifact_sample_count"),
        training_status.get("model_sample_count"),
    ]

    for candidate in candidates:
        parsed = _to_int(candidate)
        if parsed is not None:
            return parsed

    return 0


def build_model_artifact_status(
    artifact_path: Path = CALIBRATOR_PATH,
    model_dir_artifact_path: Path = MODEL_DIR_ARTIFACT_PATH,
    training_samples_path: Path = TRAINING_SAMPLES_PATH,
    training_status_path: Path = TRAINING_STATUS_PATH,
    output_path: Path = MODEL_ARTIFACT_STATUS_PATH,
    report_path: Path = REPORT_PATH,
    pipeline_version: str = PIPELINE_VERSION,
    min_clean_train_samples: int = MIN_CLEAN_TRAIN_SAMPLES,
) -> Dict[str, Any]:
    training_status, training_status_file = _read_json(training_status_path)
    training_samples_row_count, training_samples_file = _count_csv_data_rows(
        training_samples_path
    )

    selected_path = _select_artifact_path(artifact_path, model_dir_artifact_path)
    artifact_exists = selected_path.exists()
    artifact, artifact_error = _load_artifact(selected_path)

    loadable = artifact is not None
    metadata = _extract_metadata(artifact) if loadable else {}

    artifact_pipeline_version = metadata.get("pipeline_version")
    artifact_version = metadata.get("artifact_version", metadata.get("version"))
    model_type = metadata.get("model_type")
    feature_names = _extract_feature_names(metadata)

    feature_schema_hash = get_model_feature_schema_hash()
    artifact_feature_schema_hash = str(metadata.get("feature_schema_hash") or "")
    artifact_training_source = str(metadata.get("training_source") or "")
    artifact_core_feature_count = _to_int(metadata.get("core_feature_count"))
    expected_core_feature_count = len(CORE_MODEL_FEATURES)
    feature_schema_match = artifact_feature_schema_hash == feature_schema_hash

    training_status_sample_count = _to_int(training_status.get("sample_count")) or 0
    training_sample_count = _extract_training_sample_count(metadata, training_status)
    trained = bool(training_status.get("trained", False))

    sample_count_mismatch_reasons: list[str] = []

    if (
        training_samples_row_count is not None
        and training_status_sample_count > 0
        and int(training_samples_row_count) != int(training_status_sample_count)
    ):
        sample_count_mismatch_reasons.append(
            "training_samples_row_count_mismatch_training_status_sample_count"
        )

    if (
        loadable
        and training_sample_count > 0
        and training_status_sample_count > 0
        and int(training_sample_count) != int(training_status_sample_count)
    ):
        sample_count_mismatch_reasons.append(
            "artifact_training_sample_count_mismatch_training_status_sample_count"
        )

    if (
        loadable
        and training_samples_row_count is not None
        and training_sample_count > 0
        and int(training_sample_count) != int(training_samples_row_count)
    ):
        sample_count_mismatch_reasons.append(
            "artifact_training_sample_count_mismatch_training_samples_row_count"
        )

    sample_count_consistent = len(sample_count_mismatch_reasons) == 0

    valid = False
    active_model_allowed = False
    reason = "artifact_missing"
    action = "ignore_artifact_and_use_manual_baseline"

    if not artifact_exists:
        reason = "artifact_missing"
    elif not loadable:
        reason = "invalid_pickle_or_joblib_artifact"
    elif sample_count_mismatch_reasons:
        reason = "sample_count_mismatch"
    elif artifact_training_source != "data/training_samples.csv":
        reason = "training_source_mismatch"
    elif not feature_schema_match:
        reason = "feature_schema_mismatch"
    elif artifact_core_feature_count != expected_core_feature_count:
        reason = "core_feature_count_mismatch"
    elif training_sample_count < min_clean_train_samples:
        reason = "insufficient_training_samples"
    elif not artifact_pipeline_version:
        reason = "pipeline_version_missing"
    elif str(artifact_pipeline_version) != str(pipeline_version):
        reason = "pipeline_version_mismatch"
    elif not trained:
        reason = "training_status_not_trained"
    else:
        valid = True
        active_model_allowed = True
        reason = "ok"
        action = "use_artifact"

    report: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "exists": bool(artifact_exists),
        "path": str(selected_path),
        "candidate_paths": {
            "calibrator": str(artifact_path),
            "model_dir_artifact": str(model_dir_artifact_path),
        },
        "loadable": bool(loadable),
        "valid": bool(valid),
        "reason": reason,
        "action": action,
        "active_model_allowed": bool(active_model_allowed),
        "artifact_version": artifact_version,
        "pipeline_version": artifact_pipeline_version,
        "expected_pipeline_version": pipeline_version,
        "training_sample_count": int(training_sample_count),
        "training_status_sample_count": int(training_status_sample_count),
        "training_samples_row_count": (
            int(training_samples_row_count)
            if training_samples_row_count is not None
            else None
        ),
        "sample_count_consistent": bool(sample_count_consistent),
        "sample_count_mismatch_reasons": sample_count_mismatch_reasons,
        "trained": bool(trained),
        "model_type": model_type,
        "feature_schema_hash": feature_schema_hash,
        "artifact_feature_schema_hash": artifact_feature_schema_hash,
        "feature_schema_match": bool(feature_schema_match),
        "artifact_training_source": artifact_training_source,
        "artifact_core_feature_count": artifact_core_feature_count,
        "expected_core_feature_count": expected_core_feature_count,
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "metadata": metadata,
        "training_status_file": training_status_file,
        "training_samples_file": training_samples_file,
        "error": artifact_error if artifact_error else "",
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }

    _write_json(output_path, report)
    _write_json(report_path, report)
    return report


def main() -> None:
    report = build_model_artifact_status()
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
