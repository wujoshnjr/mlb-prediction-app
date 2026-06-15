from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPORT_DIR = Path("report")
DATA_DIR = Path("data")
OUTPUT_PATH = REPORT_DIR / "data_contract_report.json"

REQUIRED_JSON_REPORTS = {
    "prediction": REPORT_DIR / "prediction.json",
    "training_samples": REPORT_DIR / "training_samples_report.json",
    "training_status": DATA_DIR / "training_status.json",
    "model_artifact_status": DATA_DIR / "model_artifact_status.json",
    "model_artifact_status_report": REPORT_DIR / "model_artifact_status_report.json",
    "model_status_consistency": REPORT_DIR / "model_status_consistency_report.json",
    "artifact_rebuild_readiness": REPORT_DIR / "artifact_rebuild_readiness_report.json",
    "feature_contract": REPORT_DIR / "feature_contract_report.json",
    "feature_missingness": REPORT_DIR / "feature_missingness_report.json",
    "model_eval": REPORT_DIR / "model_eval_report.json",
    "calibration": REPORT_DIR / "calibration_report.json",
    "per_slice_performance": REPORT_DIR / "per_slice_performance_report.json",
    "outcome_linkage_diagnostic": REPORT_DIR / "outcome_linkage_diagnostic.json",
}

OPTIONAL_JSON_REPORTS = {
    "baseline_comparison": REPORT_DIR / "baseline_comparison_report.json",
    "artifact_quarantine": REPORT_DIR / "artifact_quarantine_report.json",
    "feature_priority": REPORT_DIR / "feature_priority_report.json",
    "repo_anomaly": REPORT_DIR / "repo_anomaly_report.json",
    "report_health_gate": REPORT_DIR / "report_health_gate.json",
    "clv_by_edge_bucket": REPORT_DIR / "clv_by_edge_bucket.json",
    "clv_by_side": REPORT_DIR / "clv_by_side.json",
    "clv_by_odds_range": REPORT_DIR / "clv_by_odds_range.json",
    "clv_by_lineup_status": REPORT_DIR / "clv_by_lineup_status.json",
    "walkforward": REPORT_DIR / "walkforward_evaluation.json",
    "rolling_walkforward": REPORT_DIR / "rolling_walkforward_evaluation.json",
    "lineup_starter_slice": REPORT_DIR / "lineup_starter_slice_report.json",
    "market_close": REPORT_DIR / "market_close_report.json",
    "research_quality": REPORT_DIR / "research_quality_report.json",
    "settle_reliability": REPORT_DIR / "settle_reliability_report.json",
    "settled_prediction_link": REPORT_DIR / "settled_prediction_link_report.json",
    "snapshot_sanitization": REPORT_DIR / "snapshot_sanitization_report.json",
    "prediction_sanitization": REPORT_DIR / "prediction_sanitization_report.json",
    "feature_availability": REPORT_DIR / "feature_availability_diagnostic.json",
    "feature_zero_root_cause": REPORT_DIR / "feature_zero_root_cause_diagnostic.json",
    "feature_grade": REPORT_DIR / "feature_grade_report.json",
    "daily_model_accuracy": REPORT_DIR / "daily_model_accuracy_report.json",
    "away_pick_diagnostic": REPORT_DIR / "away_pick_diagnostic_report.json",
    "away_guardrail_impact": REPORT_DIR / "away_guardrail_impact_report.json",
    "odds_fetch_diagnostic": REPORT_DIR / "odds_fetch_diagnostic.json",
    "model_lab": REPORT_DIR / "model_lab_report.json",
    "feature_promotion": REPORT_DIR / "feature_promotion_report.json",
    "finalized_linkage_diagnostic": REPORT_DIR / "finalized_linkage_diagnostic_report.json",
    "model_registry_report": REPORT_DIR / "model_registry_report.json",
    "promotion_gate": REPORT_DIR / "promotion_gate_report.json",
    "decision_audit": REPORT_DIR / "decision_audit_report.json",
    "paper_trading_ledger_report": REPORT_DIR / "paper_trading_ledger_report.json",
    "risk_exposure": REPORT_DIR / "risk_exposure_report.json",
    "artifact_retention": REPORT_DIR / "artifact_retention_manifest.json",
    "world_class_trading_system": REPORT_DIR / "world_class_trading_system_report.json",
    "saas_readiness": REPORT_DIR / "saas_readiness_report.json",
    "walk_forward_validation": REPORT_DIR / "walk_forward_validation_report.json",
    "calibration_diagnostics": REPORT_DIR / "calibration_diagnostics_report.json",
    "prediction_trust": REPORT_DIR / "prediction_trust_report.json",
    "model_comparison": REPORT_DIR / "model_comparison_report.json",
    "model_decision_guardrail": REPORT_DIR / "model_decision_guardrail_report.json",
    "shadow_ensemble_stack": REPORT_DIR / "shadow_ensemble_stack_report.json",
    "research_promotion_readiness": REPORT_DIR / "research_promotion_readiness_report.json",
    "underdog_diagnostic": REPORT_DIR / "underdog_diagnostic_report.json",
    "confidence_bucket_guardrail": REPORT_DIR / "confidence_bucket_guardrail_report.json",
    "slice_promotion_gate": REPORT_DIR / "slice_promotion_gate_report.json",
    "feature_freshness": REPORT_DIR / "feature_freshness_report.json",
    "lineup_quality": REPORT_DIR / "lineup_quality_report.json",
    "model_correctness": REPORT_DIR / "model_correctness_report.json",
    "edge_sanity_guardrail": REPORT_DIR / "edge_sanity_guardrail_report.json",
    "signal_quality": REPORT_DIR / "signal_quality_report.json",
    "product_experience": REPORT_DIR / "product_experience_report.json",
}

REQUIRED_NON_JSON_FILES = {"training_samples_csv": DATA_DIR / "training_samples.csv"}
OPTIONAL_NON_JSON_FILES = {
    "html_report": REPORT_DIR / "index.html",
    "walkforward_predictions": REPORT_DIR / "walkforward_predictions.csv",
    "rolling_walkforward_predictions": REPORT_DIR / "rolling_walkforward_predictions.csv",
    "oos_predictions": DATA_DIR / "oos_predictions_with_labels.csv",
    "decision_audit_csv": REPORT_DIR / "decision_audit.csv",
    "paper_trading_ledger_csv": DATA_DIR / "paper_trading_ledger.csv",
    "lineup_quality_context": DATA_DIR / "lineup_quality_context.csv",
    "finalized_snapshot_outcomes": DATA_DIR / "finalized_snapshot_outcomes.csv",
}

ALLOWED_STATUSES = {
    "ok",
    "warning",
    "skipped",
    "completed",
    "partial",
    "partial_failure",
    "not_attempted",
    "unavailable",
    "quarantined",
    "needs_review",
    "missing_source",
    "blocked",
    "insufficient_samples",
    "paper_trading_only_blocked_for_live",
    "insufficient_evidence",
}
FAILURE_STATUSES = {"error", "failed", "fatal"}
QUALITY_FAILURE_AS_WARNING = {"model_status_consistency"}
SAFETY_FLAGS = (
    "live_betting_allowed",
    "automated_wagering_allowed",
    "production_model_replacement_allowed",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    return value if isinstance(value, str) else str(value)


def _load_json(path: Path) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    status = {"path": str(path), "exists": path.exists(), "error": "", "type": None}
    if not path.exists():
        status["error"] = "file_missing"
        return None, status
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return None, status
    status["type"] = type(data).__name__
    if not isinstance(data, dict):
        status["error"] = "json_not_object"
        return None, status
    return data, status


def _require_keys(obj: Dict[str, Any], keys: List[str], errors: List[str], context: str) -> None:
    for key in keys:
        if key not in obj:
            errors.append(f"{context}: missing key '{key}'")


def _predictions(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = report.get("predictions") or report.get("today_predictions") or report.get("games") or []
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _contains_bad_json_scalar(value: Any, path: str = "") -> List[str]:
    hits: List[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        hits.append(path or "<root>")
    elif isinstance(value, str) and value.strip().lower() in {
        "nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }:
        hits.append(path or "<root>")
    elif isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            hits.extend(_contains_bad_json_scalar(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(_contains_bad_json_scalar(child, f"{path}[{index}]"))
    return hits


def _validate_status(name: str, report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    status = str(report.get("status", "")).strip().lower()
    if not status:
        warnings.append(f"{name}: missing status")
    elif status in FAILURE_STATUSES and name in QUALITY_FAILURE_AS_WARNING:
        warnings.append(f"{name}: governance/quality status is {status}; promotion remains locked")
    elif status in FAILURE_STATUSES:
        errors.append(f"{name}: status is {status}")
    elif status not in ALLOWED_STATUSES:
        warnings.append(f"{name}: uncommon status '{status}'")


def _validate_json_clean(name: str, report: Dict[str, Any], errors: List[str]) -> None:
    bad_paths = _contains_bad_json_scalar(report)
    if bad_paths:
        errors.append(
            f"{name}: JSON contains NaN/Infinity-like values at {', '.join(bad_paths[:12])}"
        )


def _validate_safety_flags(name: str, report: Dict[str, Any], errors: List[str]) -> None:
    for flag in SAFETY_FLAGS:
        if flag in report and report.get(flag) is not False:
            errors.append(f"{name}: {flag} must be false")


def _validate_base_report(name: str, report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    if "generated_at" not in report and "timestamp" not in report:
        warnings.append(f"{name}: missing generated_at/timestamp")
    _validate_status(name, report, errors, warnings)
    _validate_json_clean(name, report, errors)
    _validate_safety_flags(name, report, errors)


def _validate_prediction(report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    _validate_base_report("prediction", report, errors, warnings)
    predictions = _predictions(report)
    if not predictions:
        errors.append("prediction.json: no predictions found")
        return
    required = [
        "game_id",
        "recommendation",
        "model_governance_status",
        "data_quality_status",
        "live_betting_allowed",
        "live_bet_candidate",
        "stake_multiplier",
        "features",
    ]
    for index, item in enumerate(predictions[:50]):
        context = f"prediction[{index}]"
        _require_keys(item, required, errors, context)
        if item.get("live_betting_allowed") is False:
            if item.get("live_bet_candidate") is not False:
                errors.append(
                    f"{context}: live_bet_candidate must be false when live_betting_allowed is false"
                )
            try:
                stake = float(item.get("stake_multiplier") or 0.0)
            except Exception:
                stake = 999.0
            if stake != 0.0:
                errors.append(
                    f"{context}: stake_multiplier must be 0.0 when live_betting_allowed is false"
                )
        if not isinstance(item.get("model_governance_status"), dict):
            errors.append(f"{context}: model_governance_status is not an object")
        if not isinstance(item.get("data_quality_status"), dict):
            errors.append(f"{context}: data_quality_status is not an object")
        if not isinstance(item.get("features"), dict):
            errors.append(f"{context}: features must be an object")


def _validate_training_status(report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    _validate_base_report("training_status", report, errors, warnings)
    _require_keys(
        report,
        [
            "sample_count",
            "minimum_clean_train_samples",
            "trained",
            "skipped",
            "training_allowed_for_production",
        ],
        errors,
        "training_status",
    )
    try:
        sample_count = int(report.get("sample_count") or 0)
        minimum = int(report.get("minimum_clean_train_samples") or 0)
    except Exception:
        errors.append("training_status: sample_count and minimum_clean_train_samples must be numeric")
        return
    if sample_count < minimum and bool(report.get("trained", False)):
        errors.append("training_status: trained must be false when sample_count is below minimum")


def _validate_feature_contract(report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    _validate_base_report("feature_contract", report, errors, warnings)
    _require_keys(
        report,
        ["report_type", "feature_schema_hash", "core_feature_count", "checks"],
        errors,
        "feature_contract",
    )
    if str(report.get("report_type")) not in {"feature_contract_report", "feature_contract_v1"}:
        errors.append("feature_contract: unexpected report_type")
    if not isinstance(report.get("checks"), dict):
        errors.append("feature_contract: checks must be an object")


def _validate_feature_missingness(report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    _validate_base_report("feature_missingness", report, errors, warnings)
    _require_keys(
        report,
        ["report_type", "feature_schema_hash", "core_feature_count", "features", "summary"],
        errors,
        "feature_missingness",
    )
    if str(report.get("report_type")) != "feature_missingness_report":
        errors.append("feature_missingness: report_type must be feature_missingness_report")
    if not isinstance(report.get("features"), list):
        errors.append("feature_missingness: features must be a list")
    summary = report.get("summary")
    if isinstance(summary, dict):
        missing_core = summary.get("missing_core_features") or []
        if missing_core:
            errors.append(
                "feature_missingness: missing core model features: "
                + ", ".join(map(str, missing_core[:20]))
            )
    else:
        errors.append("feature_missingness: summary must be an object")


def _validate_model_eval(report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    _validate_base_report("model_eval", report, errors, warnings)
    _require_keys(
        report,
        [
            "report_type",
            "training_source",
            "features_used",
            "metrics",
            "confusion_matrix",
            *SAFETY_FLAGS,
        ],
        errors,
        "model_eval",
    )
    if str(report.get("report_type")) != "model_eval_report":
        errors.append("model_eval: report_type must be model_eval_report")
    if report.get("training_source") != "data/training_samples.csv":
        errors.append("model_eval: training_source must be data/training_samples.csv")
    if not isinstance(report.get("features_used"), list):
        errors.append("model_eval: features_used must be a list")
    if str(report.get("status", "")).lower() == "ok":
        metrics = report.get("metrics")
        if not isinstance(metrics, dict):
            errors.append("model_eval: metrics must be an object when status is ok")
        else:
            for key in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "brier",
                "logloss",
            ):
                if key not in metrics:
                    errors.append(f"model_eval.metrics: missing {key}")


def _validate_calibration(report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    _validate_base_report("calibration", report, errors, warnings)
    _require_keys(
        report,
        [
            "report_type",
            "input_path",
            "sample_count",
            "valid_sample_count",
            "brier",
            "ece",
            "mce",
            "reliability_table",
            *SAFETY_FLAGS,
        ],
        errors,
        "calibration",
    )
    if str(report.get("report_type")) != "calibration_report":
        errors.append("calibration: report_type must be calibration_report")
    if str(report.get("status", "")).lower() in {"ok", "warning"}:
        if not isinstance(report.get("reliability_table"), list):
            errors.append("calibration: reliability_table must be a list")
        for key in ("ece", "mce", "brier"):
            value = report.get(key)
            if value is None:
                errors.append(f"calibration: {key} must not be null when report is usable")
                continue
            try:
                parsed = float(value)
            except Exception:
                errors.append(f"calibration: {key} must be numeric")
                continue
            if not 0.0 <= parsed <= 1.0:
                errors.append(f"calibration: {key} must be between 0 and 1")


def _validate_per_slice_performance(report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    _validate_base_report("per_slice_performance", report, errors, warnings)
    _require_keys(
        report,
        ["report_type", "input_path", "sample_count", "valid_sample_count", "slices", *SAFETY_FLAGS],
        errors,
        "per_slice_performance",
    )
    if str(report.get("report_type")) != "per_slice_performance_report":
        errors.append("per_slice_performance: report_type must be per_slice_performance_report")
    if str(report.get("status", "")).lower() in {"ok", "warning"} and not isinstance(
        report.get("slices"), dict
    ):
        errors.append("per_slice_performance: slices must be an object when report is usable")


def _validate_generic_report(name: str, report: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    _validate_base_report(name, report, errors, warnings)


def build_contract_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    warnings: List[str] = []
    file_status: Dict[str, Any] = {}
    reports: Dict[str, Dict[str, Any]] = {}

    for name, path in REQUIRED_JSON_REPORTS.items():
        data, status = _load_json(path)
        file_status[name] = status
        if data is None:
            errors.append(f"{name}: required JSON missing or invalid ({status.get('error')})")
        else:
            reports[name] = data

    for name, path in OPTIONAL_JSON_REPORTS.items():
        data, status = _load_json(path)
        file_status[name] = status
        if data is None:
            warnings.append(f"{name}: optional JSON missing or invalid ({status.get('error')})")
        else:
            reports[name] = data

    for name, path in REQUIRED_NON_JSON_FILES.items():
        file_status[name] = {
            "path": str(path),
            "exists": path.exists(),
            "error": "" if path.exists() else "file_missing",
        }
        if not path.exists():
            errors.append(f"{name}: required file missing: {path}")

    for name, path in OPTIONAL_NON_JSON_FILES.items():
        file_status[name] = {
            "path": str(path),
            "exists": path.exists(),
            "error": "" if path.exists() else "file_missing",
        }
        if not path.exists():
            warnings.append(f"{name}: optional file missing: {path}")

    specialized = {
        "prediction": _validate_prediction,
        "training_status": _validate_training_status,
        "feature_contract": _validate_feature_contract,
        "feature_missingness": _validate_feature_missingness,
        "model_eval": _validate_model_eval,
        "calibration": _validate_calibration,
        "per_slice_performance": _validate_per_slice_performance,
    }

    for name, report in reports.items():
        validator = specialized.get(name)
        if validator is not None:
            validator(report, errors, warnings)
        else:
            _validate_generic_report(name, report, errors, warnings)

    model_eval = reports.get("model_eval") or {}
    if str(model_eval.get("status", "")).lower() == "ok" and not OPTIONAL_NON_JSON_FILES[
        "oos_predictions"
    ].exists():
        errors.append("model_eval: status ok but data/oos_predictions_with_labels.csv is missing")

    report = {
        "generated_at": _utc_now(),
        "status": "failed" if errors else "ok",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "file_status": file_status,
        "required_json_count": len(REQUIRED_JSON_REPORTS),
        "optional_json_count": len(OPTIONAL_JSON_REPORTS),
        "recommendations": []
        if not errors
        else ["Fix data contract errors before treating the pipeline output as engineering-grade."],
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }
    OUTPUT_PATH.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main() -> int:
    report = build_contract_report()
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True, allow_nan=False))
    if report["error_count"]:
        print("Data contract issues were written to report/data_contract_report.json; pipeline continues in report-only mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
