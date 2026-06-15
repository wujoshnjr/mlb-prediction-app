from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_DIR = Path("report")
OUTPUT_PATH = REPORT_DIR / "model_data_root_cause_report.json"

PREDICTION_PATH = REPORT_DIR / "prediction.json"
MODEL_EVAL_PATH = REPORT_DIR / "model_eval_report.json"
BASELINE_COMPARISON_PATH = REPORT_DIR / "baseline_comparison_report.json"
WALKFORWARD_PATH = REPORT_DIR / "walkforward_evaluation.json"
FEATURE_MISSINGNESS_PATH = REPORT_DIR / "feature_missingness_report.json"
ARTIFACT_QUARANTINE_PATH = REPORT_DIR / "artifact_quarantine_report.json"
DAILY_ACCURACY_PATH = REPORT_DIR / "daily_model_accuracy_report.json"

ZERO_RATE_BLOCK_THRESHOLD = 0.95
HIGH_ZERO_RATE_THRESHOLD = 0.50
HIGH_MISSING_RATE_THRESHOLD = 0.50


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(child) for child in value]
    return value if isinstance(value, str) else str(value)


def to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def classify_features(feature_report: dict[str, Any]) -> dict[str, Any]:
    rows = feature_report.get("features") or []
    if not isinstance(rows, list):
        rows = []

    all_zero = []
    high_zero = []
    high_missing = []
    active_core_healthy = []
    tracking_only_nonzero = []

    for item in rows:
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature") or "")
        if not feature:
            continue
        zero_rate = to_float(item.get("zero_rate")) or 0.0
        missing_rate = to_float(item.get("missing_rate")) or 0.0
        non_zero_rate = to_float(item.get("non_zero_rate")) or 0.0
        allow_main = bool(item.get("allow_in_main_model", False))
        allow_shadow = bool(item.get("allow_in_shadow_model", False))
        record = {
            "feature": feature,
            "feature_group": item.get("feature_group"),
            "zero_rate": round(zero_rate, 4),
            "missing_rate": round(missing_rate, 4),
            "non_zero_rate": round(non_zero_rate, 4),
            "allow_in_main_model": allow_main,
            "allow_in_shadow_model": allow_shadow,
            "recommended_action": item.get("recommended_action"),
        }
        if zero_rate >= ZERO_RATE_BLOCK_THRESHOLD and non_zero_rate == 0:
            all_zero.append(record)
        elif zero_rate >= HIGH_ZERO_RATE_THRESHOLD:
            high_zero.append(record)
        if missing_rate >= HIGH_MISSING_RATE_THRESHOLD:
            high_missing.append(record)
        if allow_main and zero_rate < HIGH_ZERO_RATE_THRESHOLD and missing_rate < HIGH_MISSING_RATE_THRESHOLD:
            active_core_healthy.append(record)
        if allow_shadow and not allow_main and non_zero_rate > 0:
            tracking_only_nonzero.append(record)

    key = lambda item: (-float(item.get("zero_rate", 0.0)), -float(item.get("missing_rate", 0.0)), str(item.get("feature")))
    return {
        "feature_count": len(rows),
        "active_core_healthy_count": len(active_core_healthy),
        "all_zero_count": len(all_zero),
        "high_zero_count": len(high_zero),
        "high_missing_count": len(high_missing),
        "tracking_only_nonzero_count": len(tracking_only_nonzero),
        "top_all_zero_features": sorted(all_zero, key=key)[:15],
        "top_high_zero_features": sorted(high_zero, key=key)[:15],
        "top_high_missing_features": sorted(high_missing, key=key)[:15],
        "top_tracking_only_nonzero_features": sorted(tracking_only_nonzero, key=lambda item: -float(item.get("non_zero_rate", 0.0)))[:15],
    }


def build_report() -> dict[str, Any]:
    prediction = read_json(PREDICTION_PATH)
    model_eval = read_json(MODEL_EVAL_PATH)
    baseline = read_json(BASELINE_COMPARISON_PATH)
    walkforward = read_json(WALKFORWARD_PATH)
    feature_missingness = read_json(FEATURE_MISSINGNESS_PATH)
    artifact = read_json(ARTIFACT_QUARANTINE_PATH)
    daily_accuracy = read_json(DAILY_ACCURACY_PATH)

    model_governance = nested_get(prediction, "model_governance", "model_governance") or {}
    metrics = model_eval.get("metrics") if isinstance(model_eval.get("metrics"), dict) else {}
    collapse = model_eval.get("collapse_guardrail") if isinstance(model_eval.get("collapse_guardrail"), dict) else {}
    quality_gate = baseline.get("quality_gate") if isinstance(baseline.get("quality_gate"), dict) else {}
    official_accuracy = daily_accuracy.get("official_accuracy") if isinstance(daily_accuracy.get("official_accuracy"), dict) else {}

    feature_summary = classify_features(feature_missingness)

    root_causes: list[str] = []
    if model_governance.get("is_ml_model") is False or prediction.get("model_artifact_governance", {}).get("ml_model_loaded") is False:
        root_causes.append("active_serving_is_manual_baseline_not_ml_artifact")
    if artifact.get("quarantined") is True or artifact.get("stale_sample_mismatch") is True:
        root_causes.append("model_artifact_quarantined_due_sample_mismatch")
    if collapse.get("do_not_promote") is True or collapse.get("model_has_no_discrimination_power") is True:
        root_causes.append("evaluation_model_collapse_no_discrimination_power")
    if metrics.get("predicted_positive_rate") == 1.0 or metrics.get("predicted_negative_rate") == 1.0:
        root_causes.append("single_class_prediction_collapse")
    if quality_gate.get("promotion_allowed") is False:
        root_causes.append("baseline_quality_gate_blocks_promotion")
    if walkforward.get("settled_oos_predictions") == 0:
        root_causes.append("walkforward_linkage_or_probability_extraction_empty")
    if feature_summary["all_zero_count"] > 0 or feature_summary["high_zero_count"] > 0:
        root_causes.append("many_tracking_features_are_all_zero_or_sparse")
    if feature_summary["high_missing_count"] > 0:
        root_causes.append("many_availability_or_context_features_missing_historical_coverage")

    next_actions = [
        "Keep public website labels as Projected side / Value lean / Tracking only until model collapse is fixed.",
        "Fix walkforward linkage/probability extraction first because settled OOS predictions are currently zero.",
        "Generate the standalone prediction_collapse_report.json or stop declaring it as a separate expected artifact.",
        "Backfill high-impact tracking-only features before promoting them into the active model schema.",
        "Do not load or promote the stale model artifact until artifact sample counts and model-quality gates pass.",
    ]

    report = {
        "generated_at": utc_now(),
        "report_type": "model_data_root_cause_report",
        "status": "blocked" if root_causes else "ok",
        "root_causes": sorted(set(root_causes)),
        "active_serving_state": {
            "model_source": model_governance.get("model_source"),
            "model_family": model_governance.get("model_family"),
            "is_ml_model": model_governance.get("is_ml_model"),
            "model_artifact_valid": model_governance.get("model_artifact_valid"),
            "model_artifact_allowed": model_governance.get("model_artifact_allowed"),
            "model_artifact_reason": model_governance.get("model_artifact_reason"),
            "loaded_artifact_sample_count": model_governance.get("loaded_artifact_sample_count"),
            "training_status_sample_count": model_governance.get("training_status_sample_count"),
            "clean_settled_sample_count": model_governance.get("clean_settled_sample_count"),
        },
        "model_eval_state": {
            "sample_count": model_eval.get("sample_count"),
            "test_sample_count": model_eval.get("test_sample_count"),
            "roc_auc": metrics.get("roc_auc"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "brier": metrics.get("brier"),
            "logloss": metrics.get("logloss"),
            "predicted_positive_rate": metrics.get("predicted_positive_rate"),
            "predicted_negative_rate": metrics.get("predicted_negative_rate"),
            "collapse_status": collapse.get("status"),
            "collapse_reasons": collapse.get("collapse_reasons") or [],
            "do_not_promote": collapse.get("do_not_promote"),
        },
        "baseline_gate_state": {
            "status": quality_gate.get("status"),
            "promotion_allowed": quality_gate.get("promotion_allowed"),
            "minimum_production_settled_samples": quality_gate.get("minimum_production_settled_samples"),
            "reasons": quality_gate.get("reasons") or [],
        },
        "walkforward_state": {
            "status": walkforward.get("status"),
            "prediction_snapshot_rows": nested_get(walkforward, "input_files", "prediction_snapshots", "rows"),
            "finalized_games_rows": nested_get(walkforward, "input_files", "finalized_games", "rows"),
            "total_oos_predictions": walkforward.get("total_oos_predictions"),
            "settled_oos_predictions": walkforward.get("settled_oos_predictions"),
            "walkforward_ready": walkforward.get("walkforward_ready"),
            "recommendations": walkforward.get("recommendations") or [],
        },
        "feature_data_state": feature_summary,
        "official_record_state": {
            "sample_count": official_accuracy.get("sample_count"),
            "accuracy": official_accuracy.get("accuracy"),
            "brier": official_accuracy.get("brier"),
            "logloss": official_accuracy.get("logloss"),
            "source": official_accuracy.get("source"),
        },
        "next_actions": next_actions,
        "promotion_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def main() -> int:
    report = build_report()
    print(json.dumps({"status": report["status"], "root_cause_count": len(report["root_causes"]), "output_path": str(OUTPUT_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
