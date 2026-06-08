from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

OUTPUT_PATH = REPORT_DIR / "research_promotion_readiness_report.json"

DEFAULT_INPUTS = {
    "sample_state": DATA_DIR / "sample_state.json",
    "training_status": DATA_DIR / "training_status.json",
    "finalized_linkage": REPORT_DIR / "finalized_linkage_diagnostic_report.json",
    "model_lab": REPORT_DIR / "model_lab_report.json",
    "feature_promotion": REPORT_DIR / "feature_promotion_report.json",
    "walk_forward_validation": REPORT_DIR / "walk_forward_validation_report.json",
    "calibration_diagnostics": REPORT_DIR / "calibration_diagnostics_report.json",
    "prediction_trust": REPORT_DIR / "prediction_trust_report.json",
    "model_comparison": REPORT_DIR / "model_comparison_report.json",
    "model_decision_guardrail": REPORT_DIR / "model_decision_guardrail_report.json",
    "shadow_ensemble_stack": REPORT_DIR / "shadow_ensemble_stack_report.json",
    "data_contract": REPORT_DIR / "data_contract_report.json",
    "pipeline_manifest": REPORT_DIR / "pipeline_manifest.json",
}

MIN_TRAIN_ELIGIBLE_SAMPLES = 300
MIN_PROMOTION_CLEAN_SAMPLES = 500
MIN_WALKFORWARD_OOS = 300
MAX_ECE = 0.05

UNSAFE_TRUE_FIELDS = {
    "live_betting_allowed",
    "automated_wagering_allowed",
    "user_funds_handled",
    "sportsbook_execution_enabled",
    "real_money_betting_enabled",
    "production_model_replacement_allowed",
    "production_allowed",
    "shadow_live_allowed",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}

    if isinstance(value, list):
        return [_json_safe(child) for child in value]

    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return None, status

    status["type"] = type(payload).__name__

    if not isinstance(payload, dict):
        status["error"] = "json_not_object"
        return None, status

    return payload, status


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


def _to_int(value: Any, default: int = 0) -> int:
    parsed = _to_float(value)
    if parsed is None:
        return default
    return int(parsed)


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "enabled"}

    return bool(value)


def _find_unsafe_flags(payload: Any, prefix: str = "") -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    if isinstance(payload, dict):
        for key, value in payload.items():
            current_path = f"{prefix}.{key}" if prefix else str(key)

            if key in UNSAFE_TRUE_FIELDS and _is_true(value):
                findings.append(
                    {
                        "path": current_path,
                        "field": key,
                        "value": value,
                    }
                )

            findings.extend(_find_unsafe_flags(value, current_path))

    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_find_unsafe_flags(value, f"{prefix}[{index}]"))

    return findings


def _gate(
    name: str,
    passed: bool,
    *,
    current: Any = None,
    required: Any = None,
    blocker: str = "",
    warning: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "current": current,
        "required": required,
        "blocker": "" if passed else blocker,
        "warning": warning,
        "details": details or {},
    }


def _best_shadow_model_from_model_lab(model_lab: Dict[str, Any]) -> Optional[str]:
    models = model_lab.get("models")
    if not isinstance(models, list):
        return None

    candidates = []
    for model in models:
        if not isinstance(model, dict):
            continue

        if model.get("model_name") == "market_no_vig_baseline":
            continue

        brier = _to_float(model.get("brier"))
        logloss = _to_float(model.get("logloss"))

        if brier is None:
            continue

        candidates.append(
            (
                brier,
                logloss if logloss is not None else 999.0,
                str(model.get("model_name")),
            )
        )

    if not candidates:
        return None

    candidates.sort()
    return candidates[0][2]


def _walk_forward_market_edge(walk_forward: Dict[str, Any]) -> Dict[str, Any]:
    comparison = walk_forward.get("model_vs_market")
    if not isinstance(comparison, dict):
        return {
            "any_model_beats_market_brier": False,
            "any_model_beats_market_logloss": False,
            "best_brier_delta": None,
            "best_logloss_delta": None,
            "best_brier_model": None,
            "best_logloss_model": None,
        }

    any_brier = False
    any_logloss = False
    best_brier_delta: Optional[float] = None
    best_logloss_delta: Optional[float] = None
    best_brier_model: Optional[str] = None
    best_logloss_model: Optional[str] = None

    for model_name, metrics in comparison.items():
        if not isinstance(metrics, dict):
            continue

        beats_brier = bool(metrics.get("beats_market_brier"))
        beats_logloss = bool(metrics.get("beats_market_logloss"))

        any_brier = any_brier or beats_brier
        any_logloss = any_logloss or beats_logloss

        delta_brier = _to_float(metrics.get("delta_brier"))
        delta_logloss = _to_float(metrics.get("delta_logloss"))

        if delta_brier is not None and (
            best_brier_delta is None or delta_brier < best_brier_delta
        ):
            best_brier_delta = delta_brier
            best_brier_model = str(model_name)

        if delta_logloss is not None and (
            best_logloss_delta is None or delta_logloss < best_logloss_delta
        ):
            best_logloss_delta = delta_logloss
            best_logloss_model = str(model_name)

    return {
        "any_model_beats_market_brier": any_brier,
        "any_model_beats_market_logloss": any_logloss,
        "best_brier_delta": best_brier_delta,
        "best_logloss_delta": best_logloss_delta,
        "best_brier_model": best_brier_model,
        "best_logloss_model": best_logloss_model,
    }


def _trust_summary(prediction_trust: Dict[str, Any]) -> Dict[str, Any]:
    counts = prediction_trust.get("trust_counts")
    if not isinstance(counts, dict):
        counts = {}

    total = sum(_to_int(value) for value in counts.values())
    weak = _to_int(counts.get("C")) + _to_int(counts.get("D"))
    strong = _to_int(counts.get("A")) + _to_int(counts.get("B"))

    return {
        "total": total,
        "strong_A_B": strong,
        "weak_C_D": weak,
        "weak_rate": None if total <= 0 else weak / total,
        "counts": counts,
    }


def _score_from_gates(gates: List[Dict[str, Any]]) -> float:
    if not gates:
        return 0.0

    passed = sum(1 for gate in gates if gate.get("passed"))
    return round(passed / len(gates) * 100.0, 2)


def _read_inputs(input_paths: Dict[str, Path]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    reports: Dict[str, Dict[str, Any]] = {}
    input_files: Dict[str, Dict[str, Any]] = {}

    for name, path in input_paths.items():
        payload, status = _read_json(path)
        input_files[name] = status
        reports[name] = payload or {}

    return reports, input_files


def build_report(
    *,
    output_path: Path = OUTPUT_PATH,
    input_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    paths = input_paths or DEFAULT_INPUTS

    reports, input_files = _read_inputs(paths)

    sample_state = reports.get("sample_state", {})
    training_status = reports.get("training_status", {})
    finalized_linkage = reports.get("finalized_linkage", {})
    model_lab = reports.get("model_lab", {})
    feature_promotion = reports.get("feature_promotion", {})
    walk_forward = reports.get("walk_forward_validation", {})
    calibration = reports.get("calibration_diagnostics", {})
    prediction_trust = reports.get("prediction_trust", {})
    model_comparison = reports.get("model_comparison", {})
    model_decision_guardrail = reports.get("model_decision_guardrail", {})
    shadow_ensemble = reports.get("shadow_ensemble_stack", {})
    data_contract = reports.get("data_contract", {})
    pipeline_manifest = reports.get("pipeline_manifest", {})

    warnings: List[str] = []
    blockers: List[str] = []
    next_actions: List[str] = []

    unsafe_findings: List[Dict[str, Any]] = []
    for name, report in reports.items():
        for finding in _find_unsafe_flags(report):
            finding["source"] = name
            unsafe_findings.append(finding)

    clean_settled = _to_int(
        sample_state.get("clean_settled_snapshots")
        or sample_state.get("settled_snapshots")
        or training_status.get("sample_count")
    )
    train_eligible = _to_int(
        sample_state.get("train_eligible_samples")
        or sample_state.get("clean_settled_snapshots")
        or training_status.get("sample_count")
    )

    linked_games = _to_int(
        sample_state.get("linked_games")
        or finalized_linkage.get("overlap_count_after")
    )

    pending_finalized = _to_int(finalized_linkage.get("pending_not_final_count"))
    finalized_failures = _to_int(finalized_linkage.get("api_not_found_or_failed_count"))

    model_lab_sample_count = _to_int(model_lab.get("sample_count"))
    champion_candidate = model_lab.get("champion_candidate")
    best_shadow_model = (
        model_comparison.get("recommended_challenger")
        or model_lab.get("best_by_brier")
        or _best_shadow_model_from_model_lab(model_lab)
    )

    walkforward_oos = _to_int(
        walk_forward.get("max_model_oos_predictions")
        or walk_forward.get("total_oos_predictions")
    )
    walkforward_ready = bool(walk_forward.get("walkforward_ready"))
    market_edge = _walk_forward_market_edge(walk_forward)
    beats_market = bool(
        market_edge["any_model_beats_market_brier"]
        or market_edge["any_model_beats_market_logloss"]
    )

    calibration_sample_count = _to_int(calibration.get("sample_count"))
    calibration_ready = bool(calibration.get("calibration_ready"))
    calibration_ece = _to_float(
        (calibration.get("overall") or {}).get("ece")
        if isinstance(calibration.get("overall"), dict)
        else calibration.get("ece")
    )

    feature_candidate_count = _to_int(feature_promotion.get("candidate_shadow_count"))
    feature_ready_count = _to_int(feature_promotion.get("ready_for_review_count"))

    ensemble_sample_count = _to_int(shadow_ensemble.get("sample_count"))
    ensemble_recommended = shadow_ensemble.get("recommended_shadow_ensemble")
    ensemble_promotion_eligible = bool(shadow_ensemble.get("promotion_eligible"))

    trust = _trust_summary(prediction_trust)

    data_contract_ok = data_contract.get("status") == "ok"
    pipeline_manifest_ok = bool(
        pipeline_manifest.get("tracked_file_count")
        or pipeline_manifest.get("files")
        or pipeline_manifest.get("tracked_files")
    )

    gates = [
        _gate(
            "finalized_linkage_gate",
            linked_games > 0 and finalized_failures == 0,
            current={
                "linked_games": linked_games,
                "pending_not_final": pending_finalized,
                "api_not_found_or_failed": finalized_failures,
            },
            required="linked_games > 0 and api_not_found_or_failed == 0",
            blocker="finalized linkage unavailable or API failures present",
            warning="pending games are allowed; failed finalized lookups are not",
        ),
        _gate(
            "train_sample_gate",
            train_eligible >= MIN_TRAIN_ELIGIBLE_SAMPLES,
            current=train_eligible,
            required=MIN_TRAIN_ELIGIBLE_SAMPLES,
            blocker=f"train eligible samples below threshold: {train_eligible} < {MIN_TRAIN_ELIGIBLE_SAMPLES}",
        ),
        _gate(
            "promotion_sample_gate",
            clean_settled >= MIN_PROMOTION_CLEAN_SAMPLES,
            current=clean_settled,
            required=MIN_PROMOTION_CLEAN_SAMPLES,
            blocker=f"clean settled samples below promotion threshold: {clean_settled} < {MIN_PROMOTION_CLEAN_SAMPLES}",
        ),
        _gate(
            "model_lab_gate",
            model_lab_sample_count >= MIN_TRAIN_ELIGIBLE_SAMPLES and best_shadow_model is not None,
            current={
                "sample_count": model_lab_sample_count,
                "best_shadow_model": best_shadow_model,
                "champion_candidate": champion_candidate,
            },
            required=f"sample_count >= {MIN_TRAIN_ELIGIBLE_SAMPLES} and best shadow model exists",
            blocker="model lab does not have enough samples or no challenger model",
        ),
        _gate(
            "walk_forward_oos_gate",
            walkforward_oos >= MIN_WALKFORWARD_OOS and walkforward_ready,
            current=walkforward_oos,
            required=MIN_WALKFORWARD_OOS,
            blocker=f"walk-forward OOS predictions below threshold: {walkforward_oos} < {MIN_WALKFORWARD_OOS}",
        ),
        _gate(
            "market_comparison_gate",
            beats_market,
            current=market_edge,
            required="at least one shadow model beats market brier or logloss",
            blocker="no shadow model has proven brier/logloss edge over market in walk-forward report",
        ),
        _gate(
            "model_decision_guardrail_gate",
            model_decision_guardrail.get("decision") == "NO_PROMOTION_SHADOW_ONLY"
            and not model_decision_guardrail.get("production_model_replacement_allowed", True),
            current={
                "decision": model_decision_guardrail.get("decision"),
                "status": model_decision_guardrail.get("status"),
                "recommended_challenger": model_decision_guardrail.get("recommended_challenger"),
                "probability_policy": model_decision_guardrail.get("probability_policy"),
            },
            required="guardrail must keep shadow models non-production",
            blocker="model decision guardrail missing or unsafe",
            warning="guardrail currently blocks promotion, which is expected until evidence improves",
        ),
        _gate(
            "calibration_gate",
            calibration_sample_count >= MIN_TRAIN_ELIGIBLE_SAMPLES
            and calibration_ready
            and calibration_ece is not None
            and calibration_ece <= MAX_ECE,
            current={
                "sample_count": calibration_sample_count,
                "calibration_ready": calibration_ready,
                "ece": calibration_ece,
            },
            required={
                "sample_count": MIN_TRAIN_ELIGIBLE_SAMPLES,
                "ece_max": MAX_ECE,
                "calibration_ready": True,
            },
            blocker="calibration sample count, readiness, or ECE threshold not satisfied",
        ),
        _gate(
            "feature_governance_gate",
            feature_candidate_count >= 0 and feature_ready_count == 0,
            current={
                "candidate_shadow_count": feature_candidate_count,
                "ready_for_review_count": feature_ready_count,
            },
            required="features remain in shadow unless reviewed",
            blocker="feature governance report unavailable or invalid",
            warning="ready_for_review_count is expected to remain 0 until stronger OOS evidence exists",
        ),
        _gate(
            "trust_grade_gate",
            trust["total"] >= 0,
            current=trust,
            required="prediction trust report readable",
            blocker="prediction trust report unavailable",
            warning="C/D trust grades should be monitored but do not block research readiness by themselves",
        ),
        _gate(
            "shadow_ensemble_gate",
            ensemble_sample_count == 0 or not ensemble_promotion_eligible,
            current={
                "sample_count": ensemble_sample_count,
                "recommended_shadow_ensemble": ensemble_recommended,
                "promotion_eligible": ensemble_promotion_eligible,
            },
            required="shadow ensemble must remain non-production",
            blocker="shadow ensemble claims production eligibility",
            warning="shadow ensemble is research-only and cannot replace official prediction",
        ),
        _gate(
            "data_contract_gate",
            data_contract_ok,
            current=data_contract.get("status"),
            required="ok",
            blocker="data contract report is not ok",
        ),
        _gate(
            "pipeline_manifest_gate",
            pipeline_manifest_ok,
            current={
                "tracked_file_count": pipeline_manifest.get("tracked_file_count"),
                "exists": input_files.get("pipeline_manifest", {}).get("exists"),
            },
            required="pipeline manifest exists and tracks outputs",
            blocker="pipeline manifest unavailable",
        ),
        _gate(
            "safety_lock_gate",
            len(unsafe_findings) == 0,
            current=unsafe_findings,
            required="no unsafe true flags",
            blocker="unsafe governance flag detected",
        ),
    ]

    for gate in gates:
        if not gate.get("passed") and gate.get("blocker"):
            blockers.append(str(gate["blocker"]))
        if gate.get("warning"):
            warnings.append(str(gate["warning"]))

    readiness_score = _score_from_gates(gates)

    if unsafe_findings:
        status = "failed_safety"
    elif train_eligible < MIN_TRAIN_ELIGIBLE_SAMPLES or walkforward_oos < MIN_WALKFORWARD_OOS:
        status = "insufficient_evidence"
    elif blockers:
        status = "blocked"
    else:
        status = "eligible_for_research_review"

    research_promotion_allowed = status == "eligible_for_research_review"

    if train_eligible < MIN_TRAIN_ELIGIBLE_SAMPLES:
        next_actions.append(
            f"Accumulate {MIN_TRAIN_ELIGIBLE_SAMPLES - train_eligible} more train-eligible finalized samples."
        )

    if clean_settled < MIN_PROMOTION_CLEAN_SAMPLES:
        next_actions.append(
            f"Accumulate {MIN_PROMOTION_CLEAN_SAMPLES - clean_settled} more clean settled samples for promotion-grade review."
        )

    if walkforward_oos < MIN_WALKFORWARD_OOS:
        next_actions.append(
            f"Accumulate {MIN_WALKFORWARD_OOS - walkforward_oos} more walk-forward OOS predictions."
        )

    if calibration_ece is not None and calibration_ece > MAX_ECE:
        next_actions.append(
            f"Improve calibration: ECE {calibration_ece:.4f} is above {MAX_ECE:.2f}."
        )

    if not beats_market:
        next_actions.append(
            "Require a shadow model to beat market brier or logloss over walk-forward OOS before research promotion review."
        )

    next_actions.extend(
        [
            "Keep official prediction on current calibrated logistic baseline until all gates pass.",
            "Keep LightGBM, XGBoost, residual, and ensemble outputs shadow-only.",
            "Do not lower sample gates to force promotion.",
        ]
    )

    report = {
        "generated_at": _utc_now(),
        "status": status,
        "readiness_score": readiness_score,
        "research_promotion_allowed": research_promotion_allowed,
        "official_model_replacement_allowed": False,
        "production_model_replacement_allowed": False,
        "shadow_only": True,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "user_funds_handled": False,
        "recommended_champion": champion_candidate if research_promotion_allowed else None,
        "recommended_challenger": best_shadow_model,
        "recommended_shadow_ensemble": ensemble_recommended,
        "thresholds": {
            "min_train_eligible_samples": MIN_TRAIN_ELIGIBLE_SAMPLES,
            "min_promotion_clean_samples": MIN_PROMOTION_CLEAN_SAMPLES,
            "min_walkforward_oos": MIN_WALKFORWARD_OOS,
            "max_ece": MAX_ECE,
        },
        "evidence_summary": {
            "clean_settled_samples": clean_settled,
            "train_eligible_samples": train_eligible,
            "linked_games": linked_games,
            "pending_not_final": pending_finalized,
            "model_lab_sample_count": model_lab_sample_count,
            "walkforward_oos_predictions": walkforward_oos,
            "calibration_sample_count": calibration_sample_count,
            "calibration_ece": calibration_ece,
            "feature_candidate_shadow_count": feature_candidate_count,
            "feature_ready_for_review_count": feature_ready_count,
            "prediction_trust": trust,
            "market_edge": market_edge,
            "model_decision_guardrail_status": model_decision_guardrail.get("status"),
            "model_decision_guardrail_decision": model_decision_guardrail.get("decision"),
            "probability_policy": model_decision_guardrail.get("probability_policy"),
        },
        "gates": gates,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "unsafe_governance_findings": unsafe_findings,
        "input_files": input_files,
        "next_recommended_actions": next_actions,
    }

    _write_json(output_path, report)
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
