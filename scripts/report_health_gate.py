from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from scripts.feature_schema import MODEL_FEATURES, TRACKING_ONLY_FEATURES
except Exception:
    MODEL_FEATURES = []
    TRACKING_ONLY_FEATURES = []


PREDICTION_PATH = Path("report/prediction.json")
FEATURE_AVAILABILITY_PATH = Path("report/feature_availability_diagnostic.json")
FEATURE_ZERO_ROOT_CAUSE_PATH = Path("report/feature_zero_root_cause_diagnostic.json")
FEATURE_GRADE_PATH = Path("report/feature_grade_report.json")
TRAINING_STATUS_PATH = Path("data/training_status.json")
HEALTH_GATE_REPORT_PATH = Path("report/report_health_gate.json")

MODEL_QUALITY_REPORTS = {
    "model_eval": Path("report/model_eval_report.json"),
    "prediction_collapse": Path("report/prediction_collapse_report.json"),
    "baseline_comparison": Path("report/baseline_comparison_report.json"),
    "walkforward": Path("report/walkforward_evaluation.json"),
    "artifact_quarantine": Path("report/artifact_quarantine_report.json"),
    "feature_priority": Path("report/feature_priority_report.json"),
}

EXPECTED_MODEL_TYPE = "calibrated_logistic_regression_with_imputer"
EXPECTED_MIN_CLEAN_TRAIN_SAMPLES = 300
QUALITY_BLOCK_STATUSES = {"failed", "blocked", "warning", "quarantined", "needs_review", "insufficient_samples"}
PIPELINE_FAILURE_STATUSES = {"error", "fatal"}


def _load_json(path: Path, errors: List[str]) -> Dict[str, Any]:
    if not path.exists():
        errors.append(f"Missing required JSON file: {path}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Unable to read JSON file {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        errors.append(f"JSON file is not an object: {path}")
        return {}
    return data


def _load_optional_json(path: Path, warnings: List[str]) -> Dict[str, Any]:
    if not path.exists():
        warnings.append(f"Optional quality report missing: {path}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"Unable to read optional quality report {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        warnings.append(f"Optional quality report is not an object: {path}")
        return {}
    return data


def _get_predictions(prediction_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = (
        prediction_report.get("predictions")
        or prediction_report.get("today_predictions")
        or prediction_report.get("games")
        or []
    )
    if not isinstance(candidates, list):
        return []
    return [item for item in candidates if isinstance(item, dict)]


def _contains_literal_nan(value: Any, path: str = "") -> List[str]:
    hits: List[str] = []
    if isinstance(value, str):
        if value.strip().lower() == "nan":
            hits.append(path or "<root>")
    elif isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            hits.extend(_contains_literal_nan(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            hits.extend(_contains_literal_nan(child, child_path))
    return hits


def _check_prediction_report(report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    predictions = _get_predictions(report)
    if not predictions:
        schedule_fetch_ok = report.get("schedule_fetch_ok")
        scheduled_game_count = report.get("scheduled_game_count")
        if schedule_fetch_ok is True and scheduled_game_count == 0:
            warnings.append("prediction.json has no predictions because schedule reports zero games.")
            return
        errors.append("prediction.json has no valid predictions.")
        return
    for index, prediction in enumerate(predictions):
        game_id = str(prediction.get("game_id", f"index_{index}"))
        governance = prediction.get("model_governance_status")
        if not isinstance(governance, dict):
            errors.append(f"{game_id}: missing model_governance_status.")
        else:
            if "live_betting_allowed" not in governance:
                errors.append(f"{game_id}: model_governance_status missing live_betting_allowed.")
            if "mode" not in governance:
                errors.append(f"{game_id}: model_governance_status missing mode.")
            if "block_reasons" not in governance:
                errors.append(f"{game_id}: model_governance_status missing block_reasons.")
        data_quality = prediction.get("data_quality_status")
        if not isinstance(data_quality, dict):
            errors.append(f"{game_id}: missing data_quality_status.")
        else:
            if "data_quality_grade" not in data_quality:
                errors.append(f"{game_id}: data_quality_status missing data_quality_grade.")
            if "bet_allowed" not in data_quality:
                errors.append(f"{game_id}: data_quality_status missing bet_allowed.")
            if "missing_critical_sources" not in data_quality:
                errors.append(f"{game_id}: data_quality_status missing missing_critical_sources.")
            if "missing_important_sources" not in data_quality:
                errors.append(f"{game_id}: data_quality_status missing missing_important_sources.")
            missing_important = set(data_quality.get("missing_important_sources") or [])
            if "confirmed_lineup" in missing_important or "confirmed_starter" in missing_important:
                if bool(data_quality.get("bet_allowed", False)):
                    errors.append(f"{game_id}: bet_allowed=true while lineup/starter is not confirmed.")
        live_allowed = bool(prediction.get("live_betting_allowed", False))
        live_candidate = bool(prediction.get("live_bet_candidate", False))
        stake_multiplier = float(prediction.get("stake_multiplier") or 0.0)
        if live_allowed is False and live_candidate is True:
            errors.append(f"{game_id}: live_bet_candidate=true while live_betting_allowed=false.")
        if live_allowed is False and stake_multiplier != 0.0:
            errors.append(f"{game_id}: stake_multiplier={stake_multiplier} while live_betting_allowed=false.")
        literal_nan_paths = _contains_literal_nan(prediction)
        if literal_nan_paths:
            errors.append(
                f"{game_id}: prediction contains literal 'nan' strings at: "
                + ", ".join(literal_nan_paths[:12])
            )
    generated_at = report.get("generated_at")
    if not generated_at:
        warnings.append("prediction.json missing generated_at.")


def _check_feature_availability(report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    if "non_blocking_features" not in report:
        errors.append(
            "feature_availability_diagnostic.json missing non_blocking_features. "
            "This usually means the report was generated by old diagnostic code."
        )
    high_risk_features = report.get("high_risk_features") or []
    if not isinstance(high_risk_features, list):
        errors.append("feature_availability_diagnostic.json high_risk_features is not a list.")
        return
    tracking_only = set(TRACKING_ONLY_FEATURES)
    bad_tracking_high_risk = sorted(str(feature) for feature in high_risk_features if str(feature) in tracking_only)
    if bad_tracking_high_risk:
        errors.append("Tracking-only features should not be high risk: " + ", ".join(bad_tracking_high_risk[:20]))
    if "dynamic_pythag_diff" in high_risk_features:
        errors.append("dynamic_pythag_diff is tracking-only and must not be high risk.")
    group_summary = report.get("group_summary") or {}
    if isinstance(group_summary, dict):
        for group_name, group_info in group_summary.items():
            if not isinstance(group_info, dict):
                continue
            for key in ("blocking_all_zero_count", "blocking_missing_count", "blocking_sparse_count", "non_blocking_count"):
                if key not in group_info:
                    errors.append(f"feature_availability_diagnostic group '{group_name}' missing {key}.")


def _check_feature_zero_root_cause(report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    still_zero = report.get("still_zero_features") or []
    if not isinstance(still_zero, list):
        errors.append("feature_zero_root_cause_diagnostic still_zero_features is not a list.")
        return
    still_zero_set = {str(feature) for feature in still_zero}
    model_features = set(MODEL_FEATURES)
    tracking_only_features = set(TRACKING_ONLY_FEATURES)
    blocking_zero_features = sorted(still_zero_set & model_features)
    tracking_only_zero_features = sorted(still_zero_set & tracking_only_features)
    unknown_zero_features = sorted(feature for feature in still_zero_set if feature not in model_features and feature not in tracking_only_features)
    if blocking_zero_features:
        errors.append("feature_zero_root_cause_diagnostic still has MODEL_FEATURES all-zero: " + ", ".join(blocking_zero_features[:20]))
    if unknown_zero_features:
        errors.append("feature_zero_root_cause_diagnostic still has unknown all-zero features: " + ", ".join(unknown_zero_features[:20]))
    if tracking_only_zero_features:
        warnings.append("feature_zero_root_cause_diagnostic still has tracking-only all-zero features: " + ", ".join(tracking_only_zero_features[:20]))


def _check_feature_grade(report: Dict[str, Any], errors: List[str]) -> None:
    if "grade_counts" not in report:
        errors.append("feature_grade_report.json missing grade_counts.")
    model_feature_count = int(report.get("model_feature_count") or 0)
    if MODEL_FEATURES and model_feature_count != len(MODEL_FEATURES):
        errors.append(f"feature_grade_report model_feature_count mismatch: {model_feature_count} != {len(MODEL_FEATURES)}")


def _check_training_status(report: Dict[str, Any], errors: List[str]) -> None:
    model_type = str(report.get("model_type", ""))
    if model_type != EXPECTED_MODEL_TYPE:
        errors.append(f"training_status model_type mismatch: {model_type} != {EXPECTED_MODEL_TYPE}")
    min_samples = int(report.get("minimum_clean_train_samples") or 0)
    if min_samples != EXPECTED_MIN_CLEAN_TRAIN_SAMPLES:
        errors.append(f"training_status minimum_clean_train_samples mismatch: {min_samples} != {EXPECTED_MIN_CLEAN_TRAIN_SAMPLES}")
    sample_count = int(report.get("sample_count") or 0)
    trained = bool(report.get("trained", False))
    skipped = bool(report.get("skipped", False))
    if sample_count < EXPECTED_MIN_CLEAN_TRAIN_SAMPLES:
        if trained:
            errors.append(f"training_status trained=true with insufficient samples: {sample_count} < {EXPECTED_MIN_CLEAN_TRAIN_SAMPLES}")
        if not skipped:
            errors.append(f"training_status skipped=false with insufficient samples: {sample_count} < {EXPECTED_MIN_CLEAN_TRAIN_SAMPLES}")


def _check_model_quality_reports(reports: Dict[str, Dict[str, Any]]) -> tuple[list[str], list[str]]:
    quality_blocks: list[str] = []
    notes: list[str] = []
    for name, report in reports.items():
        if not report:
            continue
        status = str(report.get("status", "")).strip().lower()
        if status in PIPELINE_FAILURE_STATUSES:
            notes.append(f"{name}: pipeline-style failure status '{status}' requires review")
            continue
        if status in QUALITY_BLOCK_STATUSES:
            quality_blocks.append(f"{name}: status={status}")
        if report.get("promotion_allowed") is False:
            quality_blocks.append(f"{name}: promotion_allowed=false")
        if report.get("do_not_promote") is True:
            quality_blocks.append(f"{name}: do_not_promote=true")
        quality_gate = report.get("quality_gate")
        if isinstance(quality_gate, dict) and quality_gate.get("promotion_allowed") is False:
            reasons = quality_gate.get("reasons") or []
            notes.append(f"{name}: quality gate blocked ({', '.join(map(str, reasons[:5]))})")
        collapse = report.get("collapse_guardrail")
        if isinstance(collapse, dict) and collapse.get("do_not_promote") is True:
            reasons = collapse.get("reasons") or []
            notes.append(f"{name}: collapse guardrail blocked ({', '.join(map(str, reasons[:5]))})")
    return sorted(set(quality_blocks)), sorted(set(notes))


def main() -> int:
    errors: List[str] = []
    warnings: List[str] = []
    prediction_report = _load_json(PREDICTION_PATH, errors)
    feature_availability = _load_json(FEATURE_AVAILABILITY_PATH, errors)
    feature_zero_root_cause = _load_json(FEATURE_ZERO_ROOT_CAUSE_PATH, errors)
    feature_grade = _load_json(FEATURE_GRADE_PATH, errors)
    training_status = _load_json(TRAINING_STATUS_PATH, errors)

    if prediction_report:
        _check_prediction_report(prediction_report, errors, warnings)
    if feature_availability:
        _check_feature_availability(feature_availability, errors, warnings)
    if feature_zero_root_cause:
        _check_feature_zero_root_cause(feature_zero_root_cause, errors, warnings)
    if feature_grade:
        _check_feature_grade(feature_grade, errors)
    if training_status:
        _check_training_status(training_status, errors)

    quality_reports = {name: _load_optional_json(path, warnings) for name, path in MODEL_QUALITY_REPORTS.items()}
    model_quality_blocks, model_quality_notes = _check_model_quality_reports(quality_reports)
    warnings.extend(model_quality_notes)

    summary = {
        "status": "failed" if errors else "ok",
        "pipeline_health_status": "failed" if errors else "ok",
        "model_quality_status": "blocked" if model_quality_blocks else "not_blocked_by_quality_reports",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "model_quality_block_count": len(model_quality_blocks),
        "errors": errors,
        "warnings": warnings,
        "model_quality_blocks": model_quality_blocks,
        "model_quality_reports_checked": sorted(MODEL_QUALITY_REPORTS.keys()),
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
        "notes": [
            "Model-quality blocks do not fail pipeline health by themselves; they keep promotion and live betting locked.",
            "Pipeline failures are reserved for invalid/missing critical files, unsafe betting flags, NaN strings, or schema violations.",
        ],
    }
    HEALTH_GATE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_GATE_REPORT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        print("Report health gate issues were written to report/report_health_gate.json; pipeline continues in report-only mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
