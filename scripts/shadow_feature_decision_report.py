from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_DIR = Path("report")
SHADOW_EXPERIMENT_PATH = REPORT_DIR / "shadow_feature_experiment_report.json"
PROMOTION_CANDIDATE_PATH = REPORT_DIR / "feature_promotion_candidate_report.json"
OUTPUT_PATH = REPORT_DIR / "shadow_feature_decision_report.json"

BARRIER_METRICS = ["brier", "logloss"]
DISPLAY_METRICS = ["accuracy", "balanced_accuracy"]
MAX_ALLOWED_BRIER_DELTA = -0.005
MAX_ALLOWED_LOGLOSS_DELTA = -0.005
MIN_RERUNS_REQUIRED = 3


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


def by_experiment(experiment_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in experiment_report.get("experiments") or []:
        if isinstance(item, dict) and item.get("experiment"):
            output[str(item["experiment"])] = item
    return output


def metric_delta(candidate: dict[str, Any], core: dict[str, Any], metric: str) -> float | None:
    c_metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    b_metrics = core.get("metrics") if isinstance(core.get("metrics"), dict) else {}
    c_value = to_float(c_metrics.get(metric))
    b_value = to_float(b_metrics.get(metric))
    if c_value is None or b_value is None:
        return None
    return round(c_value - b_value, 6)


def decide_experiment(name: str, candidate: dict[str, Any], core: dict[str, Any]) -> dict[str, Any]:
    deltas = {f"{metric}_delta_vs_core": metric_delta(candidate, core, metric) for metric in BARRIER_METRICS + DISPLAY_METRICS}
    blockers: list[str] = []
    brier_delta = deltas.get("brier_delta_vs_core")
    logloss_delta = deltas.get("logloss_delta_vs_core")
    if brier_delta is None:
        blockers.append("missing_brier_delta")
    elif brier_delta > MAX_ALLOWED_BRIER_DELTA:
        blockers.append("brier_did_not_improve_enough")
    if logloss_delta is None:
        blockers.append("missing_logloss_delta")
    elif logloss_delta > MAX_ALLOWED_LOGLOSS_DELTA:
        blockers.append("logloss_did_not_improve_enough")
    if candidate.get("status") != "ok":
        blockers.append("experiment_status_not_ok")
    sample_count = int(candidate.get("sample_count") or 0)
    if sample_count < 100:
        blockers.append("shadow_sample_count_too_small")
    decision = "reject_for_now" if blockers else "keep_for_repeated_shadow_reruns"
    return {
        "experiment": name,
        "decision": decision,
        "blockers": sorted(set(blockers)),
        "sample_count": sample_count,
        "feature_count": candidate.get("feature_count"),
        "features": candidate.get("features") or [],
        "metrics": candidate.get("metrics") or {},
        "deltas_vs_core": deltas,
    }


def build_report() -> dict[str, Any]:
    experiment_report = read_json(SHADOW_EXPERIMENT_PATH)
    candidate_report = read_json(PROMOTION_CANDIDATE_PATH)
    experiments = by_experiment(experiment_report)
    core = experiments.get("core_only", {})
    decisions = []
    for name in ("core_plus_shadow_now", "core_plus_recommended_shadow_set"):
        if name in experiments:
            decisions.append(decide_experiment(name, experiments[name], core))

    rejected = [item for item in decisions if item.get("decision") == "reject_for_now"]
    accepted_for_rerun = [item for item in decisions if item.get("decision") == "keep_for_repeated_shadow_reruns"]

    recommended_shadow_set = candidate_report.get("recommended_shadow_set") if isinstance(candidate_report.get("recommended_shadow_set"), list) else []
    shadow_now = candidate_report.get("shadow_candidate_now") if isinstance(candidate_report.get("shadow_candidate_now"), list) else []

    report = {
        "generated_at": utc_now(),
        "report_type": "shadow_feature_decision_report",
        "status": "blocked" if not accepted_for_rerun else "warning",
        "source_reports": {
            "shadow_feature_experiment": str(SHADOW_EXPERIMENT_PATH),
            "feature_promotion_candidate": str(PROMOTION_CANDIDATE_PATH),
        },
        "policy": {
            "active_model_promotion_allowed": False,
            "public_prediction_change_allowed": False,
            "minimum_required_reruns_before_any_promotion": MIN_RERUNS_REQUIRED,
            "required_brier_delta_vs_core": MAX_ALLOWED_BRIER_DELTA,
            "required_logloss_delta_vs_core": MAX_ALLOWED_LOGLOSS_DELTA,
            "reason": "Shadow features must repeatedly improve Brier and logloss before any active-model discussion. Current report is research-only.",
        },
        "core_baseline": core,
        "decisions": decisions,
        "summary": {
            "decision_count": len(decisions),
            "rejected_for_now_count": len(rejected),
            "accepted_for_repeated_shadow_rerun_count": len(accepted_for_rerun),
            "recommended_shadow_feature_count": len(recommended_shadow_set),
            "shadow_now_feature_count": len(shadow_now),
        },
        "next_actions": [
            "Do not promote shadow candidate features from the current experiment because they did not improve Brier/logloss versus core_only.",
            "Investigate whether candidate features are duplicates, stale, or too sparse before rerunning shadow experiments.",
            "Build a per-feature ablation report next to test one feature or one feature group at a time.",
            "Keep public predictions locked to existing research/tracking-only outputs.",
        ],
        "promotion_allowed": False,
        "deployment_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def main() -> int:
    report = build_report()
    print(json.dumps({"status": report["status"], "decision_count": report["summary"]["decision_count"], "output_path": str(OUTPUT_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
