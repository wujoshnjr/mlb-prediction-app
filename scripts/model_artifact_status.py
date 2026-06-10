#!/usr/bin/env python3
"""Build model artifact status.

This script must never crash simply because an artifact is missing, too small,
or unreadable. Those are expected states during evidence collection.

It writes:
- data/model_artifact_status.json
- report/model_artifact_status_report.json
"""

from __future__ import annotations

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


CALIBRATOR_PATH = Path("data/calibrator.pkl")
MODEL_DIR_ARTIFACT_PATH = Path("data/models/baseline_v2_clean/model.joblib")
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
    training_status_path: Path = TRAINING_STATUS_PATH,
    output_path: Path = MODEL_ARTIFACT_STATUS_PATH,
    report_path: Path = REPORT_PATH,
    pipeline_version: str = PIPELINE_VERSION,
    min_clean_train_samples: int = MIN_CLEAN_TRAIN_SAMPLES,
) -> Dict[str, Any]:
    training_status, training_status_file = _read_json(training_status_path)

    selected_path = _select_artifact_path(artifact_path, model_dir_artifact_path)
    artifact_exists = selected_path.exists()
    artifact, artifact_error = _load_artifact(selected_path)

    loadable = artifact is not None
    metadata = _extract_metadata(artifact) if loadable else {}

    artifact_pipeline_version = metadata.get("pipeline_version")
    artifact_version = metadata.get("artifact_version", metadata.get("version"))
    model_type = metadata.get("model_type")
    feature_names = _extract_feature_names(metadata)

    training_status_sample_count = _to_int(training_status.get("sample_count")) or 0
    training_sample_count = _extract_training_sample_count(metadata, training_status)
    trained = bool(training_status.get("trained", False))

    valid = False
    active_model_allowed = False
    reason = "artifact_missing"
    action = "ignore_artifact_and_use_manual_baseline"

    if not artifact_exists:
        reason = "artifact_missing"
    elif not loadable:
        reason = "invalid_pickle_or_joblib_artifact"
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
        "trained": bool(trained),
        "model_type": model_type,
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "metadata": metadata,
        "training_status_file": training_status_file,
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
