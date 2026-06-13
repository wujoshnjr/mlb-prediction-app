from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

OUTPUT_PATH = REPORT_DIR / "data_contract_report.json"

REQUIRED_JSON_REPORTS = {
    "prediction": REPORT_DIR / "prediction.json",
    "baseline_comparison": REPORT_DIR / "baseline_comparison_report.json",
    "clv_by_edge_bucket": REPORT_DIR / "clv_by_edge_bucket.json",
    "clv_by_side": REPORT_DIR / "clv_by_side.json",
    "clv_by_odds_range": REPORT_DIR / "clv_by_odds_range.json",
    "clv_by_lineup_status": REPORT_DIR / "clv_by_lineup_status.json",
    "calibration": REPORT_DIR / "calibration_report.json",
    "walkforward": REPORT_DIR / "walkforward_evaluation.json",
    "rolling_walkforward": REPORT_DIR / "rolling_walkforward_evaluation.json",
    "lineup_starter_slice": REPORT_DIR / "lineup_starter_slice_report.json",
    "market_close": REPORT_DIR / "market_close_report.json",
    "research_quality": REPORT_DIR / "research_quality_report.json",
    "settle_reliability": REPORT_DIR / "settle_reliability_report.json",
    "settled_prediction_link": REPORT_DIR / "settled_prediction_link_report.json",
    "snapshot_sanitization": REPORT_DIR / "snapshot_sanitization_report.json",
    "feature_availability": REPORT_DIR / "feature_availability_diagnostic.json",
    "feature_zero_root_cause": REPORT_DIR / "feature_zero_root_cause_diagnostic.json",
    "feature_grade": REPORT_DIR / "feature_grade_report.json",
    "training_status": DATA_DIR / "training_status.json",
    "sample_state": DATA_DIR / "sample_state.json",
    "sample_state_report": REPORT_DIR / "sample_state_report.json",
    "edge_sanity_guardrail": REPORT_DIR / "edge_sanity_guardrail_report.json",
    "signal_quality": REPORT_DIR / "signal_quality_report.json",
    "training_samples": REPORT_DIR / "training_samples_report.json",
    "model_artifact_status": DATA_DIR / "model_artifact_status.json",
    "model_artifact_status_report": REPORT_DIR / "model_artifact_status_report.json",
    "daily_model_accuracy": REPORT_DIR / "daily_model_accuracy_report.json",
    "away_pick_diagnostic": REPORT_DIR / "away_pick_diagnostic_report.json",
    "away_guardrail_impact": REPORT_DIR / "away_guardrail_impact_report.json",
    "odds_fetch_diagnostic": REPORT_DIR / "odds_fetch_diagnostic.json",
}

OPTIONAL_JSON_REPORTS = {
    "model_lab": REPORT_DIR / "model_lab_report.json",
    "feature_promotion": REPORT_DIR / "feature_promotion_report.json",
    "finalized_linkage_diagnostic": REPORT_DIR / "finalized_linkage_diagnostic_report.json",
    "prediction_sanitization": REPORT_DIR / "prediction_sanitization_report.json",
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
    "product_experience": REPORT_DIR / "product_experience_report.json",
}

REQUIRED_NON_JSON_FILES = {
    "html_report": REPORT_DIR / "index.html",
    "walkforward_predictions": REPORT_DIR / "walkforward_predictions.csv",
    "rolling_walkforward_predictions": REPORT_DIR / "rolling_walkforward_predictions.csv",
    "training_samples": DATA_DIR / "training_samples.csv",
}

OPTIONAL_NON_JSON_FILES = {
    "decision_audit_csv": REPORT_DIR / "decision_audit.csv",
    "paper_trading_ledger_csv": DATA_DIR / "paper_trading_ledger.csv",
    "lineup_quality_context": DATA_DIR / "lineup_quality_context.csv",
    "finalized_snapshot_outcomes": DATA_DIR / "finalized_snapshot_outcomes.csv",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _predictions(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = report.get("predictions") or report.get("today_predictions") or report.get("games") or []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _require_keys(obj: Dict[str, Any], keys: List[str], errors: List[str], context: str) -> None:
    for key in keys:
        if key not in obj:
            errors.append(f"{context}: missing key '{key}'")


def _validate_prediction(report: Dict[str, Any], errors: List[str]) -> None:
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

        data_quality = item.get("data_quality_status")
        if isinstance(data_quality, dict):
            _require_keys(
                data_quality,
                [
                    "data_quality_grade",
                    "prediction_allowed",
                    "bet_allowed",
                    "missing_critical_sources",
                    "missing_important_sources",
                ],
                errors,
                f"{context}.data_quality_status",
            )
        else:
            errors.append(f"{context}: data_quality_status is not an object")

        governance = item.get("model_governance_status")
        if isinstance(governance, dict):
            _require_keys(
                governance,
                [
                    "live_betting_allowed",
                    "mode",
                    "block_reasons",
                    "clean_model_sample_count",
                    "min_clean_train_samples",
                ],
                errors,
                f"{context}.model_governance_status",
            )
        else:
            errors.append(f"{context}: model_governance_status is not an object")


def _validate_standard_report(name: str, report: Dict[str, Any], errors: List[str]) -> None:
    _require_keys(report, ["generated_at", "status", "recommendations"], errors, name)


def _validate_baseline(report: Dict[str, Any], errors: List[str]) -> None:
    _validate_standard_report("baseline_comparison_report", report, errors)
    _require_keys(
        report,
        ["settled_prediction_count", "baselines", "comparison", "skipped_baselines"],
        errors,
        "baseline_comparison_report",
    )


def _validate_clv(name: str, report: Dict[str, Any], errors: List[str]) -> None:
    _validate_standard_report(name, report, errors)
    _require_keys(report, ["slice_type", "slices"], errors, name)

    slices = report.get("slices")
    if not isinstance(slices, list):
        errors.append(f"{name}: slices must be a list")
        return

    for index, item in enumerate(slices[:50]):
        if not isinstance(item, dict):
            errors.append(f"{name}.slices[{index}]: slice item is not object")
            continue

        _require_keys(
            item,
            [
                "slice",
                "count",
                "avg_clv",
                "positive_clv_rate",
                "paper_bet_count",
                "live_bet_candidate_count",
                "block_live_bet",
            ],
            errors,
            f"{name}.slices[{index}]",
        )


def _validate_calibration(report: Dict[str, Any], errors: List[str]) -> None:
    _validate_standard_report("calibration_report", report, errors)
    _require_keys(
        report,
        [
            "calibration_ready",
            "total_count",
            "bins",
            "weighted_ece",
            "max_calibration_error",
            "min_recommended_samples",
        ],
        errors,
        "calibration_report",
    )


def _validate_walkforward(report: Dict[str, Any], errors: List[str]) -> None:
    _validate_standard_report("walkforward_evaluation", report, errors)
    _require_keys(
        report,
        [
            "min_required_oos_predictions",
            "total_oos_predictions",
            "walkforward_ready",
            "fold_count",
        ],
        errors,
        "walkforward_evaluation",
    )


def _validate_feature_reports(reports: Dict[str, Dict[str, Any]], errors: List[str]) -> None:
    feature_availability = reports.get("feature_availability") or {}
    if "high_risk_features" not in feature_availability:
        errors.append("feature_availability_diagnostic: missing high_risk_features")
    if "non_blocking_features" not in feature_availability:
        errors.append("feature_availability_diagnostic: missing non_blocking_features")

    feature_zero = reports.get("feature_zero_root_cause") or {}
    if "still_zero_features" not in feature_zero:
        errors.append("feature_zero_root_cause_diagnostic: missing still_zero_features")

    feature_grade = reports.get("feature_grade") or {}
    if "grade_counts" not in feature_grade:
        errors.append("feature_grade_report: missing grade_counts")


def _validate_training_status(report: Dict[str, Any], errors: List[str]) -> None:
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


def _validate_daily_model_accuracy(report: Dict[str, Any], errors: List[str]) -> None:
    _validate_standard_report("daily_model_accuracy", report, errors)

    _require_keys(
        report,
        [
            "pipeline_version",
            "official_accuracy",
            "daily_accuracy",
            "rolling_windows",
            "slices",
            "pending_predictions",
            "clv_metrics",
            "prediction_probability_metrics",
            "interpretation",
            "live_betting_allowed",
            "automated_wagering_allowed",
            "production_model_replacement_allowed",
        ],
        errors,
        "daily_model_accuracy",
    )

    if report.get("live_betting_allowed") is not False:
        errors.append("daily_model_accuracy: live_betting_allowed must be false")
    if report.get("automated_wagering_allowed") is not False:
        errors.append("daily_model_accuracy: automated_wagering_allowed must be false")
    if report.get("production_model_replacement_allowed") is not False:
        errors.append("daily_model_accuracy: production_model_replacement_allowed must be false")

    official = report.get("official_accuracy")
    if isinstance(official, dict):
        _require_keys(
            official,
            ["sample_count", "correct", "accuracy", "brier", "logloss", "source"],
            errors,
            "daily_model_accuracy.official_accuracy",
        )
    else:
        errors.append("daily_model_accuracy: official_accuracy is not an object")

    clv_metrics = report.get("clv_metrics")
    if isinstance(clv_metrics, dict):
        note = str(clv_metrics.get("note") or "").lower()
        if "not win/loss accuracy" not in note:
            errors.append("daily_model_accuracy.clv_metrics: must clearly say CLV is not win/loss accuracy")
    else:
        errors.append("daily_model_accuracy: clv_metrics is not an object")

    interpretation = report.get("interpretation")
    if isinstance(interpretation, dict):
        if interpretation.get("do_not_mix_with_clv") is not True:
            errors.append("daily_model_accuracy.interpretation: do_not_mix_with_clv must be true")
    else:
        errors.append("daily_model_accuracy: interpretation is not an object")


def _validate_away_pick_diagnostic(report: Dict[str, Any], errors: List[str]) -> None:
    _validate_standard_report("away_pick_diagnostic", report, errors)

    _require_keys(
        report,
        [
            "pipeline_version",
            "report_type",
            "betting_mode",
            "input_files",
            "sample_summary",
            "official_accuracy",
            "away_segments",
            "away_by_edge_bucket",
            "away_by_market_prob_bucket",
            "clv_summary",
            "recommended_guardrails",
            "interpretation",
            "live_betting_allowed",
            "automated_wagering_allowed",
            "production_model_replacement_allowed",
        ],
        errors,
        "away_pick_diagnostic",
    )

    if report.get("report_type") != "away_pick_diagnostic_v1":
        errors.append("away_pick_diagnostic: report_type must be away_pick_diagnostic_v1")

    if report.get("betting_mode") != "paper_research":
        errors.append("away_pick_diagnostic: betting_mode must be paper_research")

    if report.get("live_betting_allowed") is not False:
        errors.append("away_pick_diagnostic: live_betting_allowed must be false")
    if report.get("automated_wagering_allowed") is not False:
        errors.append("away_pick_diagnostic: automated_wagering_allowed must be false")
    if report.get("production_model_replacement_allowed") is not False:
        errors.append("away_pick_diagnostic: production_model_replacement_allowed must be false")

    metadata = report.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("live_betting_allowed") is not False:
            errors.append("away_pick_diagnostic.metadata: live_betting_allowed must be false")
        if metadata.get("automated_wagering_allowed") is not False:
            errors.append("away_pick_diagnostic.metadata: automated_wagering_allowed must be false")
        if metadata.get("production_model_replacement_allowed") is not False:
            errors.append("away_pick_diagnostic.metadata: production_model_replacement_allowed must be false")
    else:
        errors.append("away_pick_diagnostic: metadata is not an object")

    sample_summary = report.get("sample_summary")
    if isinstance(sample_summary, dict):
        _require_keys(
            sample_summary,
            [
                "total_predictions",
                "settled_predictions",
                "pending_predictions",
                "home_pick_count",
                "away_pick_count",
                "away_pick_settled_count",
                "away_pick_pending_count",
                "away_pick_rate",
            ],
            errors,
            "away_pick_diagnostic.sample_summary",
        )
    else:
        errors.append("away_pick_diagnostic: sample_summary is not an object")

    official = report.get("official_accuracy")
    if isinstance(official, dict):
        for key in ("all_picks", "home_picks", "away_picks"):
            bucket = official.get(key)
            if isinstance(bucket, dict):
                _require_keys(
                    bucket,
                    ["sample_count", "correct", "accuracy", "brier"],
                    errors,
                    f"away_pick_diagnostic.official_accuracy.{key}",
                )
            else:
                errors.append(f"away_pick_diagnostic.official_accuracy: {key} is not an object")
    else:
        errors.append("away_pick_diagnostic: official_accuracy is not an object")

    away_segments = report.get("away_segments")
    if isinstance(away_segments, dict):
        for key in (
            "away_favorites",
            "away_underdogs",
            "away_high_edge",
            "away_low_edge",
            "away_confirmed_context",
            "away_unconfirmed_context",
        ):
            if key not in away_segments:
                errors.append(f"away_pick_diagnostic.away_segments: missing {key}")
    else:
        errors.append("away_pick_diagnostic: away_segments is not an object")

    clv_summary = report.get("clv_summary")
    if isinstance(clv_summary, dict):
        note = str(clv_summary.get("note") or "").lower()
        if "not win/loss accuracy" not in note:
            errors.append("away_pick_diagnostic.clv_summary: must clearly say CLV is not win/loss accuracy")
    else:
        errors.append("away_pick_diagnostic: clv_summary is not an object")

    interpretation = report.get("interpretation")
    if isinstance(interpretation, dict):
        _require_keys(
            interpretation,
            [
                "official_accuracy_note",
                "pending_note",
                "clv_note",
                "recommended_use",
            ],
            errors,
            "away_pick_diagnostic.interpretation",
        )
    else:
        errors.append("away_pick_diagnostic: interpretation is not an object")


def _validate_away_guardrail_impact(report: Dict[str, Any], errors: List[str]) -> None:
    _validate_standard_report("away_guardrail_impact", report, errors)

    _require_keys(
        report,
        [
            "report_type",
            "pipeline_version",
            "betting_mode",
            "input_files",
            "summary",
            "reason_counts",
            "guardrail_status_counts",
            "downgraded_examples",
            "interpretation",
            "live_betting_allowed",
            "automated_wagering_allowed",
            "production_model_replacement_allowed",
        ],
        errors,
        "away_guardrail_impact",
    )

    if report.get("report_type") != "away_guardrail_impact_v1":
        errors.append("away_guardrail_impact: report_type must be away_guardrail_impact_v1")

    if report.get("betting_mode") != "paper_research":
        errors.append("away_guardrail_impact: betting_mode must be paper_research")

    if report.get("live_betting_allowed") is not False:
        errors.append("away_guardrail_impact: live_betting_allowed must be false")
    if report.get("automated_wagering_allowed") is not False:
        errors.append("away_guardrail_impact: automated_wagering_allowed must be false")
    if report.get("production_model_replacement_allowed") is not False:
        errors.append("away_guardrail_impact: production_model_replacement_allowed must be false")

    summary = report.get("summary")
    if isinstance(summary, dict):
        _require_keys(
            summary,
            [
                "prediction_count",
                "away_candidate_count",
                "away_guardrail_applied_count",
                "away_guardrail_applied_rate",
                "retained_away_paper_signal_count",
                "downgraded_away_tracking_only_count",
                "home_candidate_count",
            ],
            errors,
            "away_guardrail_impact.summary",
        )
    else:
        errors.append("away_guardrail_impact: summary is not an object")

    if not isinstance(report.get("reason_counts"), dict):
        errors.append("away_guardrail_impact: reason_counts is not an object")

    if not isinstance(report.get("guardrail_status_counts"), dict):
        errors.append("away_guardrail_impact: guardrail_status_counts is not an object")

    if not isinstance(report.get("downgraded_examples"), list):
        errors.append("away_guardrail_impact: downgraded_examples is not a list")

    interpretation = report.get("interpretation")
    if isinstance(interpretation, dict):
        _require_keys(
            interpretation,
            [
                "guardrail_note",
                "paper_only_note",
                "recommended_use",
            ],
            errors,
            "away_guardrail_impact.interpretation",
        )
    else:
        errors.append("away_guardrail_impact: interpretation is not an object")


def _validate_odds_fetch_diagnostic(report: Dict[str, Any], errors: List[str]) -> None:
    _validate_standard_report("odds_fetch_diagnostic", report, errors)

    _require_keys(
        report,
        [
            "status",
            "sport_key",
            "endpoint",
            "requested_date",
            "attempts",
            "selected_attempt",
            "final_event_count",
            "final_usable_row_count",
            "live_betting_allowed",
            "automated_wagering_allowed",
            "production_model_replacement_allowed",
            "errors",
            "warnings",
            "recommendations",
        ],
        errors,
        "odds_fetch_diagnostic",
    )

    if report.get("live_betting_allowed") is not False:
        errors.append("odds_fetch_diagnostic: live_betting_allowed must be false")

    if report.get("automated_wagering_allowed") is not False:
        errors.append("odds_fetch_diagnostic: automated_wagering_allowed must be false")

    if report.get("production_model_replacement_allowed") is not False:
        errors.append(
            "odds_fetch_diagnostic: production_model_replacement_allowed must be false"
        )

    if not isinstance(report.get("attempts"), list):
        errors.append("odds_fetch_diagnostic: attempts must be a list")

    for index, attempt in enumerate(report.get("attempts") or []):
        if not isinstance(attempt, dict):
            errors.append(f"odds_fetch_diagnostic.attempts[{index}]: must be an object")
            continue

        _require_keys(
            attempt,
            [
                "attempt_name",
                "params",
                "status_code",
                "event_count",
                "usable_row_count",
                "quality_counts",
                "request_headers",
                "error",
            ],
            errors,
            f"odds_fetch_diagnostic.attempts[{index}]",
        )

        params = attempt.get("params")
        if isinstance(params, dict):
            api_key_value = str(params.get("apiKey") or "")
            if api_key_value and api_key_value != "***REDACTED***":
                errors.append(
                    f"odds_fetch_diagnostic.attempts[{index}]: apiKey must be redacted"
                )
        else:
            errors.append(
                f"odds_fetch_diagnostic.attempts[{index}]: params must be an object"
            )

    for numeric_key in ("final_event_count", "final_usable_row_count"):
        try:
            int(report.get(numeric_key) or 0)
        except Exception:
            errors.append(f"odds_fetch_diagnostic: {numeric_key} must be numeric")


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
            
    if "prediction" in reports:
        _validate_prediction(reports["prediction"], errors)

    if "baseline_comparison" in reports:
        _validate_baseline(reports["baseline_comparison"], errors)

    for key in (
        "clv_by_edge_bucket",
        "clv_by_side",
        "clv_by_odds_range",
        "clv_by_lineup_status",
    ):
        if key in reports:
            _validate_clv(key, reports[key], errors)

    if "calibration" in reports:
        _validate_calibration(reports["calibration"], errors)

    if "walkforward" in reports:
        _validate_walkforward(reports["walkforward"], errors)

    for standard_name in (
        "rolling_walkforward",
        "lineup_starter_slice",
        "market_close",
        "research_quality",
        "settle_reliability",
        "settled_prediction_link",
        "snapshot_sanitization",
        "model_registry_report",
        "promotion_gate",
        "decision_audit",
        "paper_trading_ledger_report",
        "risk_exposure",
        "artifact_retention",
        "world_class_trading_system",
        "saas_readiness",
        "sample_state",
        "sample_state_report",
    ):
        if standard_name in reports:
            _validate_standard_report(standard_name, reports[standard_name], errors)

    _validate_feature_reports(reports, errors)

    if "training_status" in reports:
        _validate_training_status(reports["training_status"], errors)

    if "daily_model_accuracy" in reports:
        _validate_daily_model_accuracy(reports["daily_model_accuracy"], errors)

    if "away_pick_diagnostic" in reports:
        _validate_away_pick_diagnostic(reports["away_pick_diagnostic"], errors)

    if "away_guardrail_impact" in reports:
        _validate_away_guardrail_impact(reports["away_guardrail_impact"], errors)

    if "odds_fetch_diagnostic" in reports:
        _validate_odds_fetch_diagnostic(reports["odds_fetch_diagnostic"], errors)

    report = {
        "generated_at": _utc_now(),
        "status": "failed" if errors else "ok",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "file_status": file_status,
        "recommendations": []
        if not errors
        else ["Fix data contract errors before treating the pipeline output as engineering-grade."],
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> int:
    report = build_contract_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
