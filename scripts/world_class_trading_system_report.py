from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

OUTPUT_PATH = REPORT_DIR / "world_class_trading_system_report.json"

INPUTS = {
    "prediction": REPORT_DIR / "prediction.json",
    "prediction_sanitization": REPORT_DIR / "prediction_sanitization_report.json",
    "snapshot_sanitization": REPORT_DIR / "snapshot_sanitization_report.json",
    "data_contract": REPORT_DIR / "data_contract_report.json",
    "pipeline_manifest": REPORT_DIR / "pipeline_manifest.json",
    "market_close": REPORT_DIR / "market_close_report.json",
    "evaluation_clv": REPORT_DIR / "evaluation_clv_diagnostic.json",
    "baseline_comparison": REPORT_DIR / "baseline_comparison_report.json",
    "calibration": REPORT_DIR / "calibration_report.json",
    "walkforward": REPORT_DIR / "walkforward_evaluation.json",
    "rolling_walkforward": REPORT_DIR / "rolling_walkforward_evaluation.json",
    "research_quality": REPORT_DIR / "research_quality_report.json",
    "promotion_gate": REPORT_DIR / "promotion_gate_report.json",
    "decision_audit": REPORT_DIR / "decision_audit_report.json",
    "paper_trading_ledger": REPORT_DIR / "paper_trading_ledger_report.json",
    "risk_exposure": REPORT_DIR / "risk_exposure_report.json",
    "settled_prediction_link": REPORT_DIR / "settled_prediction_link_report.json",
    "model_registry": REPORT_DIR / "model_registry_report.json",
    "feature_grade": REPORT_DIR / "feature_grade_report.json",
    "artifact_retention": REPORT_DIR / "artifact_retention_manifest.json",
    "html_report": REPORT_DIR / "index.html",
    "training_status": DATA_DIR / "training_status.json",
}


LAYER_NAMES = {
    "data_trust": "Layer 1 - Data Trust",
    "research_quality": "Layer 2 - Research Quality",
    "risk_controls": "Layer 3 - Risk Controls",
    "model_upgrade_path": "Layer 4 - Model Upgrade Path",
    "product_readiness": "Layer 5 - Product Readiness",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "error": "",
        "type": None,
    }

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


def _file_status(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "error": "" if path.exists() else "file_missing",
    }


def _load_inputs() -> Tuple[Dict[str, Optional[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    reports: Dict[str, Optional[Dict[str, Any]]] = {}
    statuses: Dict[str, Dict[str, Any]] = {}

    for name, path in INPUTS.items():
        if path.suffix.lower() == ".json":
            data, status = _read_json(path)
            reports[name] = data
            statuses[name] = status
        else:
            reports[name] = None
            statuses[name] = _file_status(path)

    return reports, statuses


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        parsed = float(value)
        if not math.isfinite(parsed):
            return None

        return parsed
    except Exception:
        return None


def _to_int(value: Any) -> int:
    parsed = _to_float(value)
    return int(parsed) if parsed is not None else 0


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "ok"}

    return bool(value)


def _status_ok(report: Optional[Dict[str, Any]]) -> bool:
    if not report:
        return False

    status = str(report.get("status") or "").strip().lower()
    return status in {"ok", "completed", "normal"}


def _predictions(prediction: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not prediction:
        return []

    raw = (
        prediction.get("today_predictions")
        or prediction.get("predictions")
        or prediction.get("games")
        or []
    )

    if not isinstance(raw, list):
        return []

    return [item for item in raw if isinstance(item, dict)]


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _stage(score: float) -> str:
    if score >= 90:
        return "institutional_readiness_candidate"
    if score >= 75:
        return "advanced_shadow_trading_candidate"
    if score >= 60:
        return "research_quality_buildout"
    if score >= 40:
        return "engineering_foundation"
    return "foundation_incomplete"


def _cap_score(score: float) -> float:
    return round(max(0.0, min(100.0, score)), 2)


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None

    return float(numerator / denominator)


def _signal(
    name: str,
    passed: bool,
    weight: float,
    details: Any = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "weight": weight,
        "details": details,
    }


def _layer_result(
    key: str,
    score: float,
    signals: List[Dict[str, Any]],
    blockers: List[str],
    warnings: List[str],
    recommendations: List[str],
) -> Dict[str, Any]:
    score = _cap_score(score)

    if blockers:
        status = "blocked"
    elif score >= 75:
        status = "ready_for_next_review"
    elif score >= 60:
        status = "building"
    else:
        status = "not_ready"

    return {
        "name": LAYER_NAMES[key],
        "status": status,
        "score": score,
        "grade": _grade(score),
        "signals": signals,
        "blockers": blockers,
        "warnings": warnings,
        "recommendations": recommendations,
    }


def _evaluate_data_trust(
    reports: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    prediction = reports.get("prediction")
    data_contract = reports.get("data_contract")
    pipeline_manifest = reports.get("pipeline_manifest")
    prediction_sanitization = reports.get("prediction_sanitization")
    snapshot_sanitization = reports.get("snapshot_sanitization")
    decision_audit = reports.get("decision_audit")
    market_close = reports.get("market_close")
    settled_link = reports.get("settled_prediction_link")

    predictions = _predictions(prediction)
    prediction_count = len(predictions)
    audit_count = _to_int((decision_audit or {}).get("audit_count"))

    sanitizer_clean = (
        prediction_sanitization is not None
        and _status_ok(prediction_sanitization)
        and _to_int(prediction_sanitization.get("change_count")) == 0
    )

    snapshot_clean = (
        snapshot_sanitization is not None
        and _status_ok(snapshot_sanitization)
        and not snapshot_sanitization.get("errors")
    )

    contract_ok = (
        data_contract is not None
        and data_contract.get("status") == "ok"
        and _to_int(data_contract.get("error_count")) == 0
    )

    manifest_readable = (
        pipeline_manifest is not None
        and _to_int(pipeline_manifest.get("invalid_json_file_count")) == 0
    )

    audit_matches_prediction = (
        prediction_count > 0
        and audit_count > 0
        and audit_count == prediction_count
    )

    clv_joined = (
        market_close is not None
        and _to_int(market_close.get("clv_available_count")) > 0
    )

    settled_link_report_ok = (
        settled_link is not None
        and _status_ok(settled_link)
        and not settled_link.get("errors")
    )
    linked_game_count = _to_int((settled_link or {}).get("linked_game_count"))

    signals = [
        _signal("data_contract_ok", contract_ok, 15, data_contract.get("status") if data_contract else None),
        _signal("pipeline_manifest_readable", manifest_readable, 15, pipeline_manifest.get("status") if pipeline_manifest else None),
        _signal("prediction_sanitizer_clean", sanitizer_clean, 15, prediction_sanitization.get("change_count") if prediction_sanitization else None),
        _signal("snapshot_sanitizer_clean", snapshot_clean, 15, snapshot_sanitization.get("status") if snapshot_sanitization else None),
        _signal("decision_audit_matches_prediction_count", audit_matches_prediction, 10, {"prediction_count": prediction_count, "audit_count": audit_count}),
        _signal("per_pick_clv_joined", clv_joined, 15, market_close.get("clv_available_count") if market_close else None),
        _signal("settled_prediction_link_report_ok", settled_link_report_ok, 15, {"linked_game_count": linked_game_count}),
    ]

    score = sum(item["weight"] for item in signals if item["passed"])
    blockers: List[str] = []
    warnings: List[str] = []

    if not prediction_count:
        blockers.append("prediction.json has no predictions")

    if not contract_ok:
        blockers.append("data contract is not ok")

    if prediction_sanitization and _to_int(prediction_sanitization.get("change_count")) > 0:
        warnings.append("prediction sanitizer still had to clean generated prediction output")

    if not clv_joined:
        warnings.append("per-pick CLV is not yet joined to market close data")

    if settled_link_report_ok and linked_game_count == 0:
        warnings.append("settled prediction link report is valid, but linked_game_count is still zero")

    return _layer_result(
        "data_trust",
        score,
        signals,
        blockers,
        warnings,
        [
            "Keep event-time, snapshot-time, entry odds, closing odds, decision audit, and settle result linked by game_id.",
            "Prediction sanitizer change_count should remain zero after prediction.py source cleanup.",
            "finalized_games.csv must be the only trusted outcome source for research evidence.",
        ],
    )
    

def _evaluate_research_quality(
    reports: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    baseline = reports.get("baseline_comparison") or {}
    market_close = reports.get("market_close") or {}
    calibration = reports.get("calibration") or {}
    walkforward = reports.get("walkforward") or {}
    rolling = reports.get("rolling_walkforward") or {}
    research = reports.get("research_quality") or {}

    comparison = baseline.get("comparison") if isinstance(baseline.get("comparison"), dict) else {}
    model_beats_market_brier = bool(comparison.get("model_beats_market_brier"))
    model_beats_market_logloss = bool(comparison.get("model_beats_market_logloss"))

    avg_clv = _to_float(market_close.get("avg_clv"))
    positive_clv_rate = _to_float(market_close.get("positive_clv_rate"))
    calibration_ready = bool(calibration.get("calibration_ready"))
    walkforward_ready = bool(walkforward.get("walkforward_ready"))
    rolling_oos = _to_int(rolling.get("total_oos_predictions"))
    settled_count = _to_int(baseline.get("settled_prediction_count"))

    clv_positive = avg_clv is not None and avg_clv > 0
    positive_clv_rate_ok = positive_clv_rate is not None and positive_clv_rate >= 0.55
    rolling_oos_ok = rolling_oos >= 300
    settled_count_ok = settled_count >= 500

    signals = [
        _signal("model_beats_market_brier", model_beats_market_brier, 15, comparison.get("model_brier")),
        _signal("model_beats_market_logloss", model_beats_market_logloss, 15, comparison.get("model_logloss")),
        _signal("avg_clv_positive", clv_positive, 20, avg_clv),
        _signal("positive_clv_rate_at_least_55pct", positive_clv_rate_ok, 15, positive_clv_rate),
        _signal("calibration_ready", calibration_ready, 10, calibration.get("total_count")),
        _signal("walkforward_ready", walkforward_ready, 10, walkforward.get("total_oos_predictions")),
        _signal("rolling_oos_at_least_300", rolling_oos_ok, 10, rolling_oos),
        _signal("settled_samples_at_least_500", settled_count_ok, 5, settled_count),
    ]

    score = sum(item["weight"] for item in signals if item["passed"])
    blockers: List[str] = []
    warnings: List[str] = []

    if not settled_count_ok:
        blockers.append(f"settled prediction samples < 500 ({settled_count})")

    if not rolling_oos_ok:
        blockers.append(f"rolling OOS predictions < 300 ({rolling_oos})")

    if not calibration_ready:
        blockers.append("calibration is not ready")

    if research.get("research_grade") in {"D", "F"}:
        blockers.append(f"research grade is too low: {research.get('research_grade')}")

    if avg_clv is None:
        warnings.append("avg_clv missing from market_close_report")
    elif avg_clv <= 0:
        warnings.append(f"avg_clv is not positive: {avg_clv}")

    return _layer_result(
        "research_quality",
        score,
        signals,
        blockers,
        warnings,
        [
            "Do not promote any model until CLV, calibration, baseline comparison, and rolling OOS evidence are all positive.",
            "Use CLV and market baseline comparison as primary research gates, not raw win rate.",
        ],
    )


def _evaluate_risk_controls(
    reports: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    prediction = reports.get("prediction")
    promotion = reports.get("promotion_gate") or {}
    paper_ledger = reports.get("paper_trading_ledger") or {}
    risk_exposure = reports.get("risk_exposure") or {}

    predictions = _predictions(prediction)

    prediction_live_flags_locked = all(
        item.get("live_betting_allowed") is False
        and item.get("live_bet_candidate") is False
        and _to_float(item.get("stake_multiplier") or 0.0) == 0.0
        for item in predictions
    ) if predictions else False

    live_stake_units = _to_float(paper_ledger.get("live_stake_units"))
    live_exposure_units = _to_float(risk_exposure.get("live_exposure_units"))
    production_allowed = bool(promotion.get("production_allowed"))
    promotion_allowed = bool(promotion.get("promotion_allowed"))
    risk_status_normal = str(risk_exposure.get("risk_status") or "").lower() in {"normal", "ok"}

    live_stake_zero = live_stake_units == 0.0
    live_exposure_zero = live_exposure_units == 0.0
    production_blocked = production_allowed is False
    promotion_not_for_live = promotion_allowed is False or production_allowed is False

    signals = [
        _signal("prediction_live_flags_locked", prediction_live_flags_locked, 25, len(predictions)),
        _signal("live_stake_units_zero", live_stake_zero, 20, live_stake_units),
        _signal("live_exposure_units_zero", live_exposure_zero, 20, live_exposure_units),
        _signal("production_not_allowed", production_blocked, 15, production_allowed),
        _signal("promotion_not_live_enabled", promotion_not_for_live, 10, promotion_allowed),
        _signal("risk_status_normal", risk_status_normal, 10, risk_exposure.get("risk_status")),
    ]

    score = sum(item["weight"] for item in signals if item["passed"])
    blockers: List[str] = []
    warnings: List[str] = []

    if not live_stake_zero:
        blockers.append(f"live stake units must be zero, got {live_stake_units}")

    if not live_exposure_zero:
        blockers.append(f"live exposure units must be zero, got {live_exposure_units}")

    if production_allowed:
        blockers.append("production live mode is enabled; this is not allowed under current governance")

    if not prediction_live_flags_locked:
        blockers.append("one or more predictions has live betting flag or non-zero stake_multiplier")

    total_open_paper_units = _to_float(risk_exposure.get("total_open_paper_units"))
    if total_open_paper_units is not None and total_open_paper_units > 10:
        warnings.append(f"paper exposure is high for research stage: {total_open_paper_units}")

    return _layer_result(
        "risk_controls",
        score,
        signals,
        blockers,
        warnings,
        [
            "Keep live betting locked until governance, sample size, CLV, calibration, and risk gates explicitly allow a later shadow phase.",
            "Paper exposure should remain small while research grade is below production readiness.",
        ],
    )


def _evaluate_model_upgrade_path(
    reports: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    training = reports.get("training_status") or {}
    model_registry = reports.get("model_registry") or {}
    feature_grade = reports.get("feature_grade") or {}
    rolling = reports.get("rolling_walkforward") or {}

    sample_count = _to_int(training.get("sample_count"))
    min_samples = _to_int(training.get("minimum_clean_train_samples"))
    trained = bool(training.get("trained"))
    registry_count = _to_int(model_registry.get("registry_count"))
    rolling_oos = _to_int(rolling.get("total_oos_predictions"))

    grade_counts = feature_grade.get("grade_counts") if isinstance(feature_grade.get("grade_counts"), dict) else {}
    a_count = _to_int(grade_counts.get("A"))
    b_count = _to_int(grade_counts.get("B"))
    total_feature_count = sum(_to_int(value) for value in grade_counts.values()) if grade_counts else 0
    ab_ratio = _safe_ratio(a_count + b_count, total_feature_count) if total_feature_count else None

    sample_progress = min(1.0, sample_count / min_samples) if min_samples > 0 else 0.0
    sample_score = 20.0 * sample_progress

    feature_quality_ok = ab_ratio is not None and ab_ratio >= 0.5
    rolling_oos_started = rolling_oos > 0

    signals = [
        _signal("training_sample_progress", sample_progress >= 1.0, 20, {"sample_count": sample_count, "minimum": min_samples}),
        _signal("model_trained", trained, 20, training.get("model_type")),
        _signal("model_registry_has_records", registry_count > 0, 20, registry_count),
        _signal("feature_ab_ratio_at_least_50pct", feature_quality_ok, 20, ab_ratio),
        _signal("rolling_oos_started", rolling_oos_started, 20, rolling_oos),
    ]

    score = sample_score
    score += sum(item["weight"] for item in signals[1:] if item["passed"])

    blockers: List[str] = []
    warnings: List[str] = []

    if sample_count < min_samples:
        blockers.append(f"clean training samples below threshold: {sample_count} < {min_samples}")

    if not trained:
        blockers.append("model is not trained yet")

    if not rolling_oos_started:
        warnings.append("rolling OOS evaluation has not started producing predictions")

    if ab_ratio is not None and ab_ratio < 0.5:
        warnings.append(f"A/B feature ratio below 50%: {ab_ratio:.4f}")

    return _layer_result(
        "model_upgrade_path",
        score,
        signals,
        blockers,
        warnings,
        [
            "Upgrade models only through registry-backed challenger evaluation.",
            "Recommended sequence: market baseline, Elo/Glicko, calibrated logistic, market residual, gradient boosting challenger, ensemble, calibration layer, risk optimizer.",
        ],
    )


def _evaluate_product_readiness(
    reports: Dict[str, Optional[Dict[str, Any]]],
    statuses: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    data_contract = reports.get("data_contract")
    pipeline_manifest = reports.get("pipeline_manifest")
    artifact_retention = reports.get("artifact_retention")
    decision_audit = reports.get("decision_audit")
    paper_ledger = reports.get("paper_trading_ledger")
    risk_exposure = reports.get("risk_exposure")
    html_status = statuses.get("html_report") or {}

    html_exists = bool(html_status.get("exists"))
    data_contract_ok = data_contract is not None and data_contract.get("status") == "ok"
    manifest_readable = pipeline_manifest is not None and _to_int(pipeline_manifest.get("invalid_json_file_count")) == 0
    artifact_retention_ok = artifact_retention is not None and artifact_retention.get("status") == "ok"
    decision_audit_ok = decision_audit is not None and _to_int(decision_audit.get("audit_count")) > 0
    ledger_ok = paper_ledger is not None and paper_ledger.get("status") == "ok"
    risk_ok = risk_exposure is not None and risk_exposure.get("status") == "ok"

    signals = [
        _signal("html_dashboard_exists", html_exists, 20, html_status.get("path")),
        _signal("data_contract_ok", data_contract_ok, 20, data_contract.get("status") if data_contract else None),
        _signal("pipeline_manifest_readable", manifest_readable, 15, pipeline_manifest.get("status") if pipeline_manifest else None),
        _signal("artifact_retention_ok", artifact_retention_ok, 15, artifact_retention.get("file_count") if artifact_retention else None),
        _signal("decision_audit_ok", decision_audit_ok, 10, decision_audit.get("audit_count") if decision_audit else None),
        _signal("paper_ledger_ok", ledger_ok, 10, paper_ledger.get("ledger_count") if paper_ledger else None),
        _signal("risk_exposure_ok", risk_ok, 10, risk_exposure.get("risk_status") if risk_exposure else None),
    ]

    score = sum(item["weight"] for item in signals if item["passed"])
    blockers: List[str] = []
    warnings: List[str] = []

    if not html_exists:
        blockers.append("dashboard HTML is missing")

    if not data_contract_ok:
        blockers.append("data contract is not ok")

    if not decision_audit_ok:
        warnings.append("decision audit is missing or empty")

    return _layer_result(
        "product_readiness",
        score,
        signals,
        blockers,
        warnings,
        [
            "Dashboard should show prediction, decision audit, CLV, promotion gate, research grade, paper PnL, and risk exposure.",
            "All product-facing output should be generated from validated reports, not ad-hoc script state.",
        ],
    )


def _overall_status(
    layers: Dict[str, Dict[str, Any]],
    critical_blockers: List[str],
    production_allowed: bool,
) -> str:
    if critical_blockers:
        return "critical_blocked"

    if production_allowed:
        return "governance_violation_production_enabled"

    if any(layer.get("status") == "blocked" for layer in layers.values()):
        return "paper_trading_only_blocked_for_live"

    return "paper_trading_only_research_mode"


def build_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reports, statuses = _load_inputs()

    layers = {
        "data_trust": _evaluate_data_trust(reports),
        "research_quality": _evaluate_research_quality(reports),
        "risk_controls": _evaluate_risk_controls(reports),
        "model_upgrade_path": _evaluate_model_upgrade_path(reports),
        "product_readiness": _evaluate_product_readiness(reports, statuses),
    }

    layer_scores = [float(layer.get("score") or 0.0) for layer in layers.values()]
    overall_score = _cap_score(sum(layer_scores) / len(layer_scores)) if layer_scores else 0.0

    promotion = reports.get("promotion_gate") or {}
    production_allowed = bool(promotion.get("production_allowed"))

    all_blockers: List[str] = []
    all_warnings: List[str] = []

    for key, layer in layers.items():
        for blocker in layer.get("blockers", []):
            all_blockers.append(f"{key}: {blocker}")
        for warning in layer.get("warnings", []):
            all_warnings.append(f"{key}: {warning}")

    critical_blockers = [
        blocker
        for blocker in all_blockers
        if "live stake" in blocker
        or "live exposure" in blocker
        or "production live mode" in blocker
        or "live betting flag" in blocker
    ]

    status = _overall_status(layers, critical_blockers, production_allowed)

    report = {
        "generated_at": _utc_now(),
        "status": status,
        "world_class_stage": _stage(overall_score),
        "overall_score": overall_score,
        "overall_grade": _grade(overall_score),
        "live_betting_allowed": False,
        "shadow_live_allowed": False,
        "production_allowed": False,
        "governance_note": "This report is audit-only. It must not enable live betting or automated bet execution.",
        "layers": layers,
        "critical_blockers": critical_blockers,
        "blockers": all_blockers,
        "warnings": all_warnings,
        "input_files": statuses,
        "recommendations": [
            "Keep live betting locked while building institutional-grade evidence.",
            "Use this report as the top-level control tower for data trust, research quality, risk controls, model upgrade path, and product readiness.",
            "Promotion must require positive CLV, market baseline outperformance, calibration readiness, rolling OOS validation, and zero live exposure governance compliance.",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
