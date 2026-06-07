from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPORT_DIR = Path("report")
DATA_DIR = Path("data")

OUTPUT_PATH = REPORT_DIR / "promotion_gate_report.json"

INPUTS = {
    "training_status": DATA_DIR / "training_status.json",
    "sample_state": DATA_DIR / "sample_state.json",
    "sample_state_report": REPORT_DIR / "sample_state_report.json",
    "baseline_comparison": REPORT_DIR / "baseline_comparison_report.json",
    "calibration": REPORT_DIR / "calibration_report.json",
    "walkforward": REPORT_DIR / "walkforward_evaluation.json",
    "rolling_walkforward": REPORT_DIR / "rolling_walkforward_evaluation.json",
    "clv_by_edge_bucket": REPORT_DIR / "clv_by_edge_bucket.json",
    "market_close": REPORT_DIR / "market_close_report.json",
    "evaluation_clv": REPORT_DIR / "evaluation_clv_diagnostic.json",
    "research_quality": REPORT_DIR / "research_quality_report.json",
    "data_contract": REPORT_DIR / "data_contract_report.json",
    "pipeline_manifest": REPORT_DIR / "pipeline_manifest.json",
}


MIN_CLEAN_SAMPLES = 500
MIN_WALKFORWARD_PREDICTIONS = 300
MIN_POSITIVE_CLV_RATE = 0.55


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    status = {"path": str(path), "exists": path.exists(), "error": ""}
    if not path.exists():
        status["error"] = "file_missing"
        return None, status

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return None, status

    if not isinstance(data, dict):
        status["error"] = "json_not_object"
        return None, status

    return data, status


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _to_int(value: Any, default: int = 0) -> int:
    parsed = _to_float(value)
    if parsed is None:
        return default
    return int(parsed)


def _large_edge_negative_clv(clv_report: Optional[Dict[str, Any]]) -> bool:
    if not clv_report:
        return False

    slices = clv_report.get("slices")
    if not isinstance(slices, list):
        return False

    for item in slices:
        if not isinstance(item, dict):
            continue

        raw_name = str(item.get("slice", "")).strip().lower()
        name = raw_name.replace("-", "_").replace(" ", "_")
        avg_clv = _to_float(item.get("avg_clv"))

        is_large_edge_bucket = (
            name == "8pct_plus"
            or name.startswith("8pct_plus")
            or name in {"large", "large_edge", "large_edge_bucket"}
        )

        if is_large_edge_bucket and avg_clv is not None and avg_clv < 0:
            return True

    return False


def _find_clv_metrics(
    market_close: Optional[Dict[str, Any]],
    evaluation_clv: Optional[Dict[str, Any]],
    walkforward: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], str]:
    for source_name, report in (
        ("market_close", market_close),
        ("evaluation_clv", evaluation_clv),
        ("walkforward", walkforward),
    ):
        if not report:
            continue

        avg_clv = _to_float(report.get("avg_clv"))
        positive_rate = _to_float(report.get("positive_clv_rate"))
        if avg_clv is not None or positive_rate is not None:
            return avg_clv, positive_rate, source_name

        clv_summary = report.get("clv_summary")
        if isinstance(clv_summary, dict):
            avg_clv = _to_float(clv_summary.get("avg_clv"))
            positive_rate = _to_float(clv_summary.get("positive_clv_rate"))
            if avg_clv is not None or positive_rate is not None:
                return avg_clv, positive_rate, f"{source_name}.clv_summary"

    return None, None, "missing"


def build_report() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reports: Dict[str, Optional[Dict[str, Any]]] = {}
    input_files: Dict[str, Any] = {}
    warnings: List[str] = []
    errors: List[str] = []
    blockers: List[str] = []
    evaluated: Dict[str, Any] = {}

    for name, path in INPUTS.items():
        data, status = _read_json(path)
        reports[name] = data
        input_files[name] = status
        if data is None:
            warnings.append(f"Missing or invalid input report: {path}")

    training = reports.get("training_status") or {}
    sample_state = reports.get("sample_state") or reports.get("sample_state_report") or {}
    baseline = reports.get("baseline_comparison") or {}
    calibration = reports.get("calibration") or {}
    walkforward = reports.get("rolling_walkforward") or reports.get("walkforward") or {}
    clv_edge = reports.get("clv_by_edge_bucket")
    market_close = reports.get("market_close")
    evaluation_clv = reports.get("evaluation_clv")
    research = reports.get("research_quality") or {}
    data_contract = reports.get("data_contract") or {}

    clean_samples = _to_int(
        sample_state.get("clean_settled_snapshots")
        or sample_state.get("train_eligible_samples")
        or training.get("sample_count")
        or training.get("clean_model_sample_count")
        or baseline.get("settled_prediction_count"),
        0,
    )
    evaluated["sample_state_source"] = "data/sample_state.json"
    evaluated["sample_state_training_allowed"] = sample_state.get("training_allowed")
    evaluated["sample_state_promotion_sample_ready"] = sample_state.get("promotion_sample_ready")
    if clean_samples < MIN_CLEAN_SAMPLES:
        blockers.append("clean settled samples < 500")

    wf_predictions = _to_int(
        sample_state.get("walkforward_predictions")
        or walkforward.get("total_oos_predictions")
        or walkforward.get("walkforward_predictions"),
        0,
    )
    evaluated["walkforward_predictions"] = wf_predictions
    evaluated["min_walkforward_predictions"] = MIN_WALKFORWARD_PREDICTIONS
    if wf_predictions < MIN_WALKFORWARD_PREDICTIONS:
        blockers.append("walk-forward predictions < 300")

    model_brier = _to_float(walkforward.get("model_brier"))
    market_brier = _to_float(walkforward.get("market_brier"))
    evaluated["model_brier"] = model_brier
    evaluated["market_brier"] = market_brier
    if model_brier is None or market_brier is None:
        blockers.append("missing model/market brier comparison")
    elif model_brier >= market_brier:
        blockers.append("model_brier is not better than market_brier")

    model_logloss = _to_float(walkforward.get("model_logloss"))
    market_logloss = _to_float(walkforward.get("market_logloss"))
    evaluated["model_logloss"] = model_logloss
    evaluated["market_logloss"] = market_logloss
    if model_logloss is None or market_logloss is None:
        blockers.append("missing model/market logloss comparison")
    elif model_logloss >= market_logloss:
        blockers.append("model_logloss is not better than market_logloss")

    avg_clv, positive_clv_rate, clv_signal_source = _find_clv_metrics(
        market_close,
        evaluation_clv,
        walkforward,
    )
    evaluated["avg_clv"] = avg_clv
    evaluated["positive_clv_rate"] = positive_clv_rate
    evaluated["clv_signal_source"] = clv_signal_source

    if avg_clv is None:
        blockers.append("avg_clv missing")
    elif avg_clv <= 0:
        blockers.append("avg_clv <= 0")

    if positive_clv_rate is None:
        blockers.append("positive_clv_rate missing")
    elif positive_clv_rate <= MIN_POSITIVE_CLV_RATE:
        blockers.append("positive_clv_rate <= 0.55")

    calibration_ready = bool(calibration.get("calibration_ready"))
    evaluated["calibration_ready"] = calibration_ready
    if not calibration_ready:
        blockers.append("calibration_ready is false")

    data_contract_ok = data_contract.get("status") == "ok"
    evaluated["data_contract_ok"] = data_contract_ok
    if not data_contract_ok:
        blockers.append("data_contract_report status is not ok")

    if _large_edge_negative_clv(clv_edge):
        blockers.append("large-edge bucket has negative CLV")

    research_grade = research.get("research_grade")
    evaluated["research_grade"] = research_grade
    if research_grade in {"D", "F"}:
        blockers.append("research quality grade is D/F")

    blockers.append("live betting disabled by governance")

    if clean_samples < MIN_CLEAN_SAMPLES or wf_predictions < MIN_WALKFORWARD_PREDICTIONS:
        status = "insufficient_samples"
    elif blockers:
        status = "blocked"
    else:
        status = "eligible"

    promotion_allowed = status == "eligible" and not blockers
    shadow_live_allowed = False
    production_allowed = False

    report = {
        "generated_at": _utc_now(),
        "status": status,
        "input_files": input_files,
        "promotion_allowed": promotion_allowed,
        "shadow_live_allowed": shadow_live_allowed,
        "production_allowed": production_allowed,
        "blockers": blockers,
        "warnings": warnings,
        "errors": errors,
        "evaluated_conditions": evaluated,
        "recommendations": [
            "Promotion gate is audit-only. Live betting remains disabled unless governance is explicitly changed."
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
