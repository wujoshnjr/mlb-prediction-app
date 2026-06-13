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
    safe = _json_safe(data)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            safe,
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
        status["error"] = "file not found"
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
        status["error"] = "file not found"
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


def _metadata_value(source: dict[str, Any], key: str) -> Any:
    if key in source:
        return source.get(key)

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
        return info, "artifact file not found"

    artifact: Any = None

    if joblib is not None:
        try:
            artifact = joblib.load(path)
        except Exception:
            artifact = None

    if artifact is None:
        try:
            with path.open("rb") as handle:
                artifact = pickle.load(handle)
        except Exception as exc:
            return info, str(exc)

    try:
        if isinstance(artifact, dict):
            artifact_metadata = artifact.get("metadata", {})
            if not isinstance(artifact_metadata, dict):
                artifact_metadata = {}

            for key in metadata_keys:
                value = _metadata_value(artifact, key)
                if value is None:
                    value = _metadata_value(artifact_metadata, key)
                info[key] = value

            return info, ""

        artifact_metadata = getattr(artifact, "metadata", {})
        if not isinstance(artifact_metadata, dict):
            artifact_metadata = {}

        for key in metadata_keys:
            if hasattr(artifact, key):
                info[key] = getattr(artifact, key)
            elif key in artifact_metadata:
                info[key] = artifact_metadata.get(key)

        return info, ""

    except Exception as exc:
        return info, str(exc)


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
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue

            prediction_governance = prediction.get("model_governance")
            if isinstance(prediction_governance, dict):
                values.append(
                    prediction_governance.get("loaded_artifact_sample_count")
                )

            prediction_governance_status = prediction.get("model_governance_status")
            if isinstance(prediction_governance_status, dict):
                values.append(
                    prediction_governance_status.get("loaded_artifact_sample_count")
                )

    return unique_ints(values)


def extract_prediction_active_model_allowed(
    prediction_report: dict[str, Any],
) -> bool:
    if not isinstance(prediction_report, dict):
        return False

    if prediction_report.get("active_model_allowed") is True:
        return True

    for key in ("model_governance", "model_governance_status"):
        value = prediction_report.get(key)
        if isinstance(value, dict) and value.get("active_model_allowed") is True:
            return True

    predictions = prediction_report.get("predictions")
    if isinstance(predictions, list):
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue

            for key in ("model_governance", "model_governance_status"):
                value = prediction.get(key)
                if isinstance(value, dict) and value.get("active_model_allowed") is True:
                    return True

    return False


def _compare_positive_ints(
    mismatches: list[dict[str, Any]],
    field: str,
    left: int | None,
    right: int | None,
) -> None:
    if left is None or right is None:
        return

    if left <= 0 or right <= 0:
        return

    if int(left) != int(right):
        mismatches.append(
            {
                "field": field,
                "left": int(left),
                "right": int(right),
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
    report["input_files"]["training_samples"] = training_samples_status
    report["training_samples_row_count"] = training_samples_row_count

    training_status_payload, training_status_file = read_json_safe(
        TRAINING_STATUS_PATH
    )
    report["input_files"]["training_status"] = training_status_file

    if isinstance(training_status_payload, dict):
        report["training_status_sample_count"] = to_int(
            training_status_payload.get("sample_count")
        )
        report["training_status_trained"] = bool(
            training_status_payload.get("trained", False)
        )
        report["training_status_reason"] = str(
            training_status_payload.get("reason")
            or training_status_payload.get("skip_reason")
            or ""
        )

    artifact_status_payload, artifact_status_file = read_json_safe(
        MODEL_ARTIFACT_STATUS_PATH
    )
    report["input_files"]["model_artifact_status"] = artifact_status_file

    if isinstance(artifact_status_payload, dict):
        report["artifact_status_training_sample_count"] = to_int(
            artifact_status_payload.get("training_sample_count")
        )
        report["artifact_status_training_status_sample_count"] = to_int(
            artifact_status_payload.get("training_status_sample_count")
        )
        report["artifact_status_training_samples_row_count"] = to_int(
            artifact_status_payload.get("training_samples_row_count")
        )

        sample_count_consistent = artifact_status_payload.get(
            "sample_count_consistent"
        )
        if isinstance(sample_count_consistent, bool):
            report["artifact_status_sample_count_consistent"] = (
                sample_count_consistent
            )

        if artifact_status_payload.get("active_model_allowed") is True:
            report["active_model_allowed"] = True

    prediction_report_payload, prediction_report_file = read_json_safe(
        PREDICTION_REPORT_PATH
    )
    report["input_files"]["prediction_report"] = prediction_report_file

    if isinstance(prediction_report_payload, dict):
        report["prediction_loaded_artifact_sample_counts"] = (
            extract_prediction_loaded_artifact_sample_counts(
                prediction_report_payload
            )
        )

        if extract_prediction_active_model_allowed(prediction_report_payload):
            report["active_model_allowed"] = True

    artifact_path = select_artifact_path()
    artifact_info, artifact_error = load_artifact_metadata(artifact_path)

    report["artifact_selected_path"] = str(artifact_path)
    report["artifact_exists"] = artifact_path.exists()
    report["artifact_loadable"] = (
        artifact_path.exists() and artifact_error == ""
    )
    report["artifact_metadata_training_sample_count"] = to_int(
        artifact_info.get("training_sample_count")
    )
    report["artifact_metadata_pipeline_version"] = artifact_info.get(
        "pipeline_version"
    )

    report["input_files"]["artifact"] = {
        "path": str(artifact_path),
        "exists": artifact_path.exists(),
        "loadable": report["artifact_loadable"],
        "error": artifact_error,
    }

    mismatches: list[dict[str, Any]] = []

    training_samples_row_count = report["training_samples_row_count"]
    training_status_sample_count = report["training_status_sample_count"]
    artifact_status_training_sample_count = report[
        "artifact_status_training_sample_count"
    ]
    artifact_metadata_training_sample_count = report[
        "artifact_metadata_training_sample_count"
    ]

    _compare_positive_ints(
        mismatches,
        "training_samples_vs_training_status",
        training_samples_row_count,
        training_status_sample_count,
    )
    _compare_positive_ints(
        mismatches,
        "artifact_status_vs_training_status",
        artifact_status_training_sample_count,
        training_status_sample_count,
    )
    _compare_positive_ints(
        mismatches,
        "artifact_metadata_vs_training_status",
        artifact_metadata_training_sample_count,
        training_status_sample_count,
    )
    _compare_positive_ints(
        mismatches,
        "artifact_metadata_vs_training_samples",
        artifact_metadata_training_sample_count,
        training_samples_row_count,
    )

    for loaded_artifact_sample_count in report[
        "prediction_loaded_artifact_sample_counts"
    ]:
        _compare_positive_ints(
            mismatches,
            "prediction_loaded_artifact_sample_count_vs_training_status",
            loaded_artifact_sample_count,
            training_status_sample_count,
        )

    if report["artifact_status_sample_count_consistent"] is False:
        mismatches.append(
            {
                "field": "model_artifact_status_sample_count_consistent",
                "left": False,
                "right": True,
            }
        )

    report["mismatches"] = mismatches
    report["sample_count_consistent"] = len(mismatches) == 0

    report["trained"] = bool(
        report["training_status_trained"]
        and report["artifact_exists"]
        and report["artifact_loadable"]
        and report["sample_count_consistent"]
    )

    file_issue = False
    for status in report["input_files"].values():
        if not isinstance(status, dict):
            continue

        if status.get("error") or not bool(status.get("exists", True)):
            file_issue = True
            break

    if mismatches:
        report["status"] = "failed"
    elif file_issue:
        report["status"] = "partial"
    else:
        report["status"] = "ok"

    if mismatches:
        report["recommendations"].append(
            "Do not activate model artifact until sample-count mismatches are resolved."
        )

    if not report["artifact_exists"]:
        report["recommendations"].append(
            "No model artifact exists; manual baseline should remain active."
        )

    if not report["active_model_allowed"]:
        report["recommendations"].append(
            "Active model is not allowed; prediction.py should use manual baseline."
        )

    if (
        training_status_sample_count is not None
        and training_status_sample_count > 0
        and training_status_sample_count < 300
    ):
        report["recommendations"].append(
            "Training sample count is below minimum production training threshold."
        )

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
