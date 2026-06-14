from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAINING_SAMPLES_PATH = Path("data/training_samples.csv")
TRAINING_STATUS_PATH = Path("data/training_status.json")
MODEL_ARTIFACT_STATUS_PATH = Path("data/model_artifact_status.json")
MODEL_ARTIFACT_STATUS_REPORT_PATH = Path("report/model_artifact_status_report.json")
MODEL_STATUS_CONSISTENCY_REPORT_PATH = Path("report/model_status_consistency_report.json")
MODEL_LAB_REPORT_PATH = Path("report/model_lab_report.json")
WALK_FORWARD_VALIDATION_REPORT_PATH = Path("report/walk_forward_validation_report.json")
REPORT_PATH = Path("report/artifact_rebuild_readiness_report.json")

REPORT_TYPE = "artifact_rebuild_readiness_v1"

MIN_CLEAN_TRAIN_SAMPLES = 300
MIN_PROMOTION_SAMPLES = 500
MIN_WALK_FORWARD_OOS = 300


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


def read_json_safe(path: Path) -> tuple[Any, dict[str, Any]]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        " path.parent.mkdir(parents=True, exist_ok=True)
    safe_data = _json_safe(data)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            safe_data,
            handle,
            indent=2,
           error": "",
        "type": "",
    }

    if not path.exists():
        status["error"] = "file not found"
        return None, status

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        payload = _json_safe(payload)

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
    """Count data rows in a CSV file, excluding the header row."""
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


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None

        parsed = float(value)
        if not math.isfinite(parsed):
            return None

        return parsed

    except Exception:
        return None


def bool_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return False


def append_blocker(
    blockers: list[dict[str, str]],
    code: str,
    detail: str,
) -> None:
    blockers.append(
        {
            "code": code,
            "detail": detail,
        }
    )


def _input_has_issue(input_files: dict[str, Any]) -> bool:
    for status in input_files.values():
        if not isinstance(status, dict):
            continue

        if status.get("error") or not bool(status.get("exists", True)):
            return True

    return False


def _max_model_oos(model_oos_counts: Any) -> int | None:
    if not isinstance(model_oos_counts, dict) or not model_oos_counts:
        return None

    values: list[int] = []

    for value in model_oos_counts.values():
        parsed = to_int(value)
        if parsed is not None:
            values.append(parsed)

    if not values:
        return None

    return max(values)


def generate_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": _utc_now(),
        "status": "ok",
        "report_type": REPORT_TYPE,
        "input_files": {},
        "training_samples_row_count": None,
        "training_status_sample_count": None,
        "training_status_trained": False,
        "model_artifact_valid": False,
        "model_artifact_reason": "",
        "model_artifact_action": "",
        "model_status_consistency_status": "",
        "model_status_consistency_sample_count_consistent": False,
        "model_status_consistency_mismatch_count": 0,
        "model_lab_sample_count": None,
        "model_lab_best_by_brier": "",
        "model_lab_best_by_logloss": "",
        "model_lab_best_by_ece": "",
        "walk_forward_ready": False,
        "walk_forward_total_oos_predictions": None,
        "walk_forward_unique_oos_games": None,
        "walk_forward_max_model_oos": None,
        "minimum_clean_train_samples": MIN_CLEAN_TRAIN_SAMPLES,
        "minimum_promotion_samples": MIN_PROMOTION_SAMPLES,
        "minimum_walk_forward_oos": MIN_WALK_FORWARD_OOS,
        "artifact_rebuild_allowed": False,
        "artifact_rebuild_status": "blocked",
        "artifact_quarantine_required": False,
        "promotion_candidate_allowed": False,
        "production_model_replacement_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "blockers": [],
        "warnings": [],
        "recommendations": [],
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
        report["training_status_trained"] = bool_or_false(
            training_status_payload.get("trained")
        )

    model_artifact_payload, model_artifact_file = read_json_safe(
        MODEL_ARTIFACT_STATUS_PATH
    )
    report["input_files"]["model_artifact_status"] = model_artifact_file

    if isinstance(model_artifact_payload, dict):
        report["model_artifact_valid"] = bool_or_false(
            model_artifact_payload.get("valid")
        )
        report["model_artifact_reason"] = str(
            model_artifact_payload.get("reason") or ""
        )
        report["model_artifact_action"] = str(
            model_artifact_payload.get("action") or ""
        )

    _, model_artifact_status_report_file = read_json_safe(
        MODEL_ARTIFACT_STATUS_REPORT_PATH
    )
    report["input_files"]["model_artifact_status_report"] = (
        model_artifact_status_report_file
    )

    model_status_consistency_payload, model_status_consistency_file = read_json_safe(
        MODEL_STATUS_CONSISTENCY_REPORT_PATH
    )
    report["input_files"]["model_status_consistency"] = model_status_consistency_file

    if isinstance(model_status_consistency_payload, dict):
        report["model_status_consistency_status"] = str(
            model_status_consistency_payload.get("status") or ""
        )
        report["model_status_consistency_sample_count_consistent"] = bool_or_false(
            model_status_consistency_payload.get("sample_count_consistent")
        )

        mismatches = model_status_consistency_payload.get("mismatches")
        if isinstance(mismatches, list):
            report["model_status_consistency_mismatch_count"] = len(mismatches)
        else:
            report["model_status_consistency_mismatch_count"] = 0

    model_lab_payload, model_lab_file = read_json_safe(MODEL_LAB_REPORT_PATH)
    report["input_files"]["model_lab"] = model_lab_file

    if isinstance(model_lab_payload, dict):
        report["model_lab_sample_count"] = to_int(
            model_lab_payload.get("sample_count")
        )
        report["model_lab_best_by_brier"] = str(
            model_lab_payload.get("best_by_brier") or ""
        )
        report["model_lab_best_by_logloss"] = str(
            model_lab_payload.get("best_by_logloss") or ""
        )
        report["model_lab_best_by_ece"] = str(
            model_lab_payload.get("best_by_ece") or ""
        )

    walk_forward_payload, walk_forward_file = read_json_safe(
        WALK_FORWARD_VALIDATION_REPORT_PATH
    )
    report["input_files"]["walk_forward_validation"] = walk_forward_file

    if isinstance(walk_forward_payload, dict):
        report["walk_forward_ready"] = bool_or_false(
            walk_forward_payload.get("walkforward_ready")
        )
        report["walk_forward_total_oos_predictions"] = to_int(
            walk_forward_payload.get("total_oos_predictions")
        )
        report["walk_forward_unique_oos_games"] = to_int(
            walk_forward_payload.get("unique_oos_games")
        )
        report["walk_forward_max_model_oos"] = _max_model_oos(
            walk_forward_payload.get("model_oos_counts")
        )

    blockers: list[dict[str, str]] = []

    if report["training_samples_row_count"] is None:
        append_blocker(
            blockers,
            "training_samples_missing",
            "training_samples.csv is missing or unreadable.",
        )

    if report["training_status_sample_count"] is None:
        append_blocker(
            blockers,
            "training_status_missing",
            "training_status.json is missing or unreadable.",
        )

    if (
        report["training_samples_row_count"] is not None
        and report["training_samples_row_count"] < MIN_CLEAN_TRAIN_SAMPLES
    ):
        append_blocker(
            blockers,
            "insufficient_training_samples_for_rebuild",
            (
                f"Only {report['training_samples_row_count']} training samples, "
                f"need at least {MIN_CLEAN_TRAIN_SAMPLES}."
            ),
        )

    if (
        report["training_status_sample_count"] is not None
        and report["training_status_sample_count"] < MIN_CLEAN_TRAIN_SAMPLES
    ):
        append_blocker(
            blockers,
            "insufficient_training_status_samples_for_rebuild",
            (
                f"Only {report['training_status_sample_count']} training status samples, "
                f"need at least {MIN_CLEAN_TRAIN_SAMPLES}."
            ),
        )

    if (
        report["training_samples_row_count"] is not None
        and report["training_status_sample_count"] is not None
        and report["training_samples_row_count"]
        != report["training_status_sample_count"]
    ):
        append_blocker(
            blockers,
            "training_samples_mismatch_training_status",
            (
                f"CSV rows {report['training_samples_row_count']} != "
                f"training_status sample_count {report['training_status_sample_count']}."
            ),
        )

    if report["model_status_consistency_mismatch_count"] > 0:
        append_blocker(
            blockers,
            "model_status_consistency_mismatches_present",
            (
                "Model status consistency report contains "
                f"{report['model_status_consistency_mismatch_count']} mismatch(es)."
            ),
        )

    if report["walk_forward_max_model_oos"] is None:
        append_blocker(
            blockers,
            "walk_forward_oos_missing",
            "Walk-forward per-model OOS count is missing.",
        )

    elif report["walk_forward_max_model_oos"] < MIN_WALK_FORWARD_OOS:
        append_blocker(
            blockers,
            "insufficient_walk_forward_oos_for_readiness",
            (
                f"Max model OOS is {report['walk_forward_max_model_oos']}, "
                f"need at least {MIN_WALK_FORWARD_OOS}."
            ),
        )

    report["blockers"] = blockers

    artifact_quarantine_required = False

    if (
        report["model_artifact_valid"] is False
        and report["model_artifact_reason"] != ""
    ):
        artifact_quarantine_required = True

    if report["model_status_consistency_mismatch_count"] > 0:
        artifact_quarantine_required = True

    if report["model_artifact_reason"] == "sample_count_mismatch":
        artifact_quarantine_required = True

    report["artifact_quarantine_required"] = artifact_quarantine_required

    input_issue = _input_has_issue(report["input_files"])

    if blockers:
        report["status"] = "blocked"
        report["artifact_rebuild_status"] = "blocked"
    elif input_issue:
        report["status"] = "partial"
        report["artifact_rebuild_status"] = "partial"
    else:
        report["status"] = "ok"
        report["artifact_rebuild_status"] = "ready_for_rebuild"

    report["artifact_rebuild_allowed"] = bool(
        not blockers
        and report["training_samples_row_count"] is not None
        and report["training_status_sample_count"] is not None
        and report["training_samples_row_count"] >= MIN_CLEAN_TRAIN_SAMPLES
        and report["training_status_sample_count"] >= MIN_CLEAN_TRAIN_SAMPLES
        and report["training_samples_row_count"]
        == report["training_status_sample_count"]
        and report["model_status_consistency_sample_count_consistent"] is True
        and report["model_status_consistency_mismatch_count"] == 0
        and report["live_betting_allowed"] is False
        and report["automated_wagering_allowed"] is False
        and report["production_model_replacement_allowed"] is False
    )

    report["promotion_candidate_allowed"] = bool(
        report["artifact_rebuild_allowed"] is True
        and report["training_samples_row_count"] is not None
        and report["training_samples_row_count"] >= MIN_PROMOTION_SAMPLES
        and report["walk_forward_ready"] is True
        and report["walk_forward_max_model_oos"] is not None
        and report["walk_forward_max_model_oos"] >= MIN_WALK_FORWARD_OOS
        and report["model_status_consistency_mismatch_count"] == 0
    )

    recommendations: list[str] = []

    if report["artifact_quarantine_required"]:
        recommendations.append(
            "Keep current artifact quarantined until rebuilt from current canonical training samples."
        )

    if report["artifact_rebuild_allowed"] is False:
        recommendations.append(
            "Do not rebuild artifact until readiness blockers are resolved."
        )

    if (
        report["training_samples_row_count"] is not None
        and report["training_samples_row_count"] < MIN_CLEAN_TRAIN_SAMPLES
    ):
        recommendations.append(
            "Continue collecting finalized samples until at least 300 clean training samples are available."
        )

    if (
        report["training_samples_row_count"] is not None
        and report["training_samples_row_count"] < MIN_PROMOTION_SAMPLES
    ):
        recommendations.append(
            "Do not consider promotion until at least 500 clean finalized samples are available."
        )

    recommendations.append("Production model replacement remains disabled by policy.")

    report["recommendations"] = recommendations

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
