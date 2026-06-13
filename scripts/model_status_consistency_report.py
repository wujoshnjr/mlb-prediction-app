from __future__ import annotations

import csv
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import joblib
except Exception:
    joblib = None


TRAINING_SAMPLES_PATH = Path("data/training_samples.csv")
TRAINING_STATUS_PATH = Path("data/training_status.json")
CALIBRATOR_PATH = Path("data/calibrator.pkl")
MODEL_DIR_ARTIFACT_PATH = Path("data/models/baseline_v2_clean/model.joblib")
MODEL_ARTIFACT_STATUS_PATH = Path("data/model_artifact_status.json")
PREDICTION_REPORT_PATH = Path("report/prediction.json")
REPORT_PATH = Path("report/model_status_consistency_report.json")
REPORT_TYPE = "model_status_consistency_v1"
MIN_PRODUCTION_TRAINING_SAMPLES = 300


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (datetime,)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

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

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    return str(value)


def safe_json_dump(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(data),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def read_json_safe(path: Path) -> tuple[Any, dict[str, Any]]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "error": "",
        "type": "",
    }

    if not path.exists():
        return None, status

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            status["type"] = "dict"
        elif isinstance(payload, list):
            status["type"] = "list"
        else:
            status["type"] = type(payload).__name__

        return payload, status

    except Exception as exc:
        status["error"] = str(exc)
        return None, status


def count_csv_rows(path: Path) -> tuple[int | None, dict[str, Any]]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "rows": None,
        "error": "",
    }

    if not path.exists():
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


def to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None

        parsed = float(value)
        if not math.isfinite(parsed):
            return None

        return int(parsed)

    except Exception:
        return None


def unique_ints(values: list[Any]) -> list[int]:
    output: set[int] = set()

    for value in values:
        parsed = to_int(value)
        if parsed is not None:
            output.add(parsed)

    return sorted(output)


def _pick_first_int(mapping: dict[str, Any], keys: list[str]) -> int | None:
    for key in keys:
        parsed = to_int(mapping.get(key))
        if parsed is not None:
            return parsed

    return None


def _pick_first_value(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value

    return None


def load_artifact_metadata(path: Path) -> tuple[dict[str, Any], str]:
    metadata_keys = [
        "training_sample_count",
        "sample_count",
        "n_samples",
        "pipeline_version",
        "artifact_version",
        "version",
        "model_type",
        "training_source",
    ]

    info: dict[str, Any] = {key: None for key in metadata_keys}

    if not path.exists():
        return info, ""

    artifact: Any = None
    load_errors: list[str] = []

    if joblib is not None:
        try:
            artifact = joblib.load(path)
        except Exception as exc:
            load_errors.append(f"joblib: {exc}")

    if artifact is None:
        try:
            with path.open("rb") as handle:
                artifact = pickle.load(handle)
        except Exception as exc:
            load_errors.append(f"pickle: {exc}")
            return info, "; ".join(load_errors)

    try:
        if isinstance(artifact, dict):
            artifact_metadata = artifact.get("metadata", {})
            if not isinstance(artifact_metadata, dict):
                artifact_metadata = {}

            for key in metadata_keys:
                if artifact.get(key) is not None:
                    info[key] = artifact.get(key)
                elif artifact_metadata.get(key) is not None:
                    info[key] = artifact_metadata.get(key)

        else:
            artifact_metadata = getattr(artifact, "metadata", {})
            if not isinstance(artifact_metadata, dict):
                artifact_metadata = {}

            for key in metadata_keys:
                if hasattr(artifact, key):
                    info[key] = getattr(artifact, key)
                elif artifact_metadata.get(key) is not None:
                    info[key] = artifact_metadata.get(key)

    except Exception as exc:
        return info, str(exc)

    return info, ""


def select_artifact_path() -> Path:
    if MODEL_DIR_ARTIFACT_PATH.exists():
        return MODEL_DIR_ARTIFACT_PATH

    return CALIBRATOR_PATH


def extract_prediction_loaded_artifact_sample_counts(
    prediction_report: dict[str, Any],
) -> list[int]:
    values: list[Any] = []

    if not isinstance(prediction_report, dict):
        return []

    model_governance = prediction_report.get("model_governance")
    if isinstance(model_governance, dict):
        values.append(model_governance.get("loaded_artifact_sample_count"))

    model_governance_status = prediction_report.get("model_governance_status")
    if isinstance(model_governance_status, dict):
        values.append(model_governance_status.get("loaded_artifact_sample_count"))

    predictions = prediction_report.get("predictions")
    if isinstance(predictions, list):
        for item in predictions:
            if not isinstance(item, dict):
                continue

            item_governance_status = item.get("model_governance_status")
            if isinstance(item_governance_status, dict):
                values.append(
                    item_governance_status.get("loaded_artifact_sample_count")
                )

            item_governance = item.get("model_governance")
            if isinstance(item_governance, dict):
                values.append(item_governance.get("loaded_artifact_sample_count"))

    return unique_ints(values)


def extract_prediction_active_model_allowed(prediction_report: dict[str, Any]) -> bool:
    if not isinstance(prediction_report, dict):
        return False

    direct_value = prediction_report.get("active_model_allowed")
    if isinstance(direct_value, bool):
        return direct_value

    for key in ["model_governance", "model_governance_status"]:
        section = prediction_report.get(key)
        if isinstance(section, dict) and isinstance(section.get("active_model_allowed"), bool):
            return bool(section.get("active_model_allowed"))

    predictions = prediction_report.get("predictions")
    if isinstance(predictions, list):
        for item in predictions:
            if not isinstance(item, dict):
                continue

            for key in ["model_governance", "model_governance_status"]:
                section = item.get(key)
                if (
                    isinstance(section, dict)
                    and isinstance(section.get("active_model_allowed"), bool)
                ):
                    return bool(section.get("active_model_allowed"))

    return False


def _numeric_compare_should_run(left: int | None, right: int | None) -> bool:
    return left is not None and right is not None and left > 0 and right > 0


def _add_mismatch_if_needed(
    mismatches: list[dict[str, Any]],
    field: str,
    left: int | None,
    right: int | None,
) -> None:
    if _numeric_compare_should_run(left, right) and left != right:
        mismatches.append(
            {
                "field": field,
                "left": left,
                "right": right,
            }
        )


def generate_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": _utc_now(),
        "status": "ok",
        "report_type": REPORT_TYPE,
        "input_files": {},
        "training_samples_row_count": None,
        "training_status_sample_count": None,
        "training_status_trained": False,
        "training_status_reason": "",
        "artifact_selected_path": "",
        "artifact_exists": False,
        "artifact_loadable": False,
        "artifact_metadata_training_sample_count": None,
        "artifact_metadata_pipeline_version": None,
        "artifact_status_training_sample_count": None,
        "artifact_status_training_status_sample_count": None,
        "artifact_status_training_samples_row_count": None,
        "artifact_status_sample_count_consistent": None,
        "prediction_loaded_artifact_sample_counts": [],
        "active_model_allowed": False,
        "trained": False,
        "sample_count_consistent": True,
        "mismatches": [],
        "warnings": [],
        "errors": [],
        "recommendations": [],
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }

    training_samples_row_count, training_samples_status = count_csv_rows(
        TRAINING_SAMPLES_PATH
    )
    training_status_payload, training_status_file = read_json_safe(
        TRAINING_STATUS_PATH
    )
    model_artifact_status_payload, model_artifact_status_file = read_json_safe(
        MODEL_ARTIFACT_STATUS_PATH
    )
    prediction_payload, prediction_file = read_json_safe(PREDICTION_REPORT_PATH)

    artifact_path = select_artifact_path()
    artifact_metadata, artifact_error = load_artifact_metadata(artifact_path)
    artifact_exists = artifact_path.exists()
    artifact_loadable = artifact_exists and artifact_error == ""

    report["input_files"] = {
        "training_samples": training_samples_status,
        "training_status": training_status_file,
        "model_artifact_status": model_artifact_status_file,
        "prediction_report": prediction_file,
        "artifact": {
            "path": str(artifact_path),
            "exists": artifact_exists,
            "loadable": artifact_loadable,
            "error": artifact_error,
        },
    }

    report["training_samples_row_count"] = training_samples_row_count

    if isinstance(training_status_payload, dict):
        report["training_status_sample_count"] = _pick_first_int(
            training_status_payload,
            ["sample_count", "training_sample_count", "clean_training_rows"],
        )
        report["training_status_trained"] = bool(
            training_status_payload.get("trained", False)
        )
        report["training_status_reason"] = str(
            _pick_first_value(
                training_status_payload,
                ["reason", "skip_reason", "status_reason"],
            )
            or ""
        )

    if isinstance(model_artifact_status_payload, dict):
        report["artifact_status_training_sample_count"] = to_int(
            model_artifact_status_payload.get("training_sample_count")
        )
        report["artifact_status_training_status_sample_count"] = to_int(
            model_artifact_status_payload.get("training_status_sample_count")
        )
        report["artifact_status_training_samples_row_count"] = to_int(
            model_artifact_status_payload.get("training_samples_row_count")
        )

        sample_count_consistent = model_artifact_status_payload.get(
            "sample_count_consistent"
        )
        if isinstance(sample_count_consistent, bool):
            report["artifact_status_sample_count_consistent"] = sample_count_consistent

        if isinstance(model_artifact_status_payload.get("active_model_allowed"), bool):
            report["active_model_allowed"] = bool(
                model_artifact_status_payload.get("active_model_allowed")
            )

    if isinstance(prediction_payload, dict):
        report["prediction_loaded_artifact_sample_counts"] = (
            extract_prediction_loaded_artifact_sample_counts(prediction_payload)
        )

        prediction_active_allowed = extract_prediction_active_model_allowed(
            prediction_payload
        )
        report["active_model_allowed"] = (
            bool(report["active_model_allowed"]) or prediction_active_allowed
        )

    report["artifact_selected_path"] = str(artifact_path)
    report["artifact_exists"] = bool(artifact_exists)
    report["artifact_loadable"] = bool(artifact_loadable)
    report["artifact_metadata_training_sample_count"] = to_int(
        artifact_metadata.get("training_sample_count")
        or artifact_metadata.get("sample_count")
        or artifact_metadata.get("n_samples")
    )
    report["artifact_metadata_pipeline_version"] = artifact_metadata.get(
        "pipeline_version"
    )

    mismatches: list[dict[str, Any]] = []

    training_samples_count = report["training_samples_row_count"]
    training_status_count = report["training_status_sample_count"]
    artifact_status_training_count = report["artifact_status_training_sample_count"]
    artifact_status_training_status
