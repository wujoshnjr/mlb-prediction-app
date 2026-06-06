from __future__ import annotations

import json
import sys
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

EXPECTED_MODEL_TYPE = "calibrated_logistic_regression_with_imputer"
EXPECTED_MIN_CLEAN_TRAIN_SAMPLES = 300


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
            if (
                "confirmed_lineup" in missing_important
                or "confirmed_starter" in missing_important
            ):
                if bool(data_quality.get("bet_allowed", False)):
                    errors.append(
                        f"{game_id}: bet_allowed=true while lineup/starter is not confirmed."
                    )

        live_allowed = bool(prediction.get("live_betting_allowed", False))
        live_candidate = bool(prediction.get("live_bet_candidate", False))
        stake_multiplier = float(prediction.get("stake_multiplier") or 0.0)

        if live_allowed is False and live_candidate is True:
            errors.append(f"{game_id}: live_bet_candidate=true while live_betting_allowed=false.")

        if live_allowed is False and stake_multiplier != 0.0:
            errors.append(
                f"{game_id}: stake_multiplier={stake_multiplier} while live_betting_allowed=false."
            )

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
    bad_tracking_high_risk = sorted(
        str(feature) for feature in high_risk_features if str(feature) in tracking_only
    )

    if bad_tracking_high_risk:
        errors.append(
            "Tracking-only features should not be high risk: "
            + ", ".join(bad_tracking_high_risk[:20])
        )

    if "dynamic_pythag_diff" in high_risk_features:
        errors.append("dynamic_pythag_diff is tracking-only and must not be high risk.")

    group_summary = report.get("group_summary") or {}
    if isinstance(group_summary, dict):
        for group_name, group_info in group_summary.items():
            if not isinstance(group_info, dict):
                continue

            for key in (
                "blocking_all_zero_count",
                "blocking_missing_count",
                "blocking_sparse_count",
                "non_blocking_count",
            ):
                if key not in group_info:
                    errors.append(
                        f"feature_availability_diagnostic group '{group_name}' missing {key}."
                    )


def _check_feature_zero_root_cause(report: Dict[str, Any], errors: List[str]) -> None:
    still_zero = report.get("still_zero_features") or []

    if not isinstance(still_zero, list):
        errors.append("feature_zero_root_cause_diagnostic still_zero_features is not a list.")
        return

    if still_zero:
        errors.append(
            "feature_zero_root_cause_diagnostic still has zero features: "
            + ", ".join(str(feature) for feature in still_zero[:20])
        )


def _check_feature_grade(report: Dict[str, Any], errors: List[str]) -> None:
    if "grade_counts" not in report:
        errors.append("feature_grade_report.json missing grade_counts.")

    model_feature_count = int(report.get("model_feature_count") or 0)
    if MODEL_FEATURES and model_feature_count != len(MODEL_FEATURES):
        errors.append(
            f"feature_grade_report model_feature_count mismatch: "
            f"{model_feature_count} != {len(MODEL_FEATURES)}"
        )


def _check_training_status(report: Dict[str, Any], errors: List[str]) -> None:
    model_type = str(report.get("model_type", ""))
    if model_type != EXPECTED_MODEL_TYPE:
        errors.append(
            f"training_status model_type mismatch: {model_type} != {EXPECTED_MODEL_TYPE}"
        )

    min_samples = int(report.get("minimum_clean_train_samples") or 0)
    if min_samples != EXPECTED_MIN_CLEAN_TRAIN_SAMPLES:
        errors.append(
            f"training_status minimum_clean_train_samples mismatch: "
            f"{min_samples} != {EXPECTED_MIN_CLEAN_TRAIN_SAMPLES}"
        )

    sample_count = int(report.get("sample_count") or 0)
    trained = bool(report.get("trained", False))
    skipped = bool(report.get("skipped", False))

    if sample_count < EXPECTED_MIN_CLEAN_TRAIN_SAMPLES:
        if trained:
            errors.append(
                f"training_status trained=true with insufficient samples: "
                f"{sample_count} < {EXPECTED_MIN_CLEAN_TRAIN_SAMPLES}"
            )
        if not skipped:
            errors.append(
                f"training_status skipped=false with insufficient samples: "
                f"{sample_count} < {EXPECTED_MIN_CLEAN_TRAIN_SAMPLES}"
            )


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
        _check_feature_zero_root_cause(feature_zero_root_cause, errors)

    if feature_grade:
        _check_feature_grade(feature_grade, errors)

    if training_status:
        _check_training_status(training_status, errors)

    summary = {
        "status": "failed" if errors else "ok",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
