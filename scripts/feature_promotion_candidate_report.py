from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_DIR = Path("report")
FEATURE_SOURCE_COVERAGE_PATH = REPORT_DIR / "feature_source_coverage_report.json"
WALKFORWARD_PATH = REPORT_DIR / "walkforward_evaluation.json"
MODEL_ROOT_CAUSE_PATH = REPORT_DIR / "model_data_root_cause_report.json"
BASELINE_COMPARISON_PATH = REPORT_DIR / "baseline_comparison_report.json"
SHADOW_ABLATION_PATH = REPORT_DIR / "shadow_feature_ablation_report.json"
OUTPUT_PATH = REPORT_DIR / "feature_promotion_candidate_report.json"

SHADOW_CANDIDATE_CATEGORIES = {
    "feature_disabled_with_signal",
    "feature_disabled_with_partial_signal",
    "tracking_coverage_ok",
}

BLOCKED_CATEGORIES = {
    "source_missing_all_zero",
    "source_missing_or_not_backfilled",
    "all_zero_needs_review",
    "high_zero_needs_review",
}

MIN_SHADOW_NON_ZERO_RATE = 0.30
MAX_SHADOW_MISSING_RATE = 0.10
MIN_DIRECT_SHADOW_NON_ZERO_RATE = 0.45
MAX_DIRECT_SHADOW_ZERO_RATE = 0.60
MAX_CANDIDATES_PER_GROUP = 4
MAX_TOTAL_CANDIDATES = 18

GROUP_WEIGHTS = {
    "starting_pitcher": 10.0,
    "statcast_batting": 9.0,
    "lineup": 9.0,
    "bullpen": 8.0,
    "market": 8.0,
    "core_strength": 6.0,
    "rest_travel": 5.5,
    "park_weather": 5.0,
    "availability": 3.5,
    "ungrouped": 2.0,
}


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


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def collect_feature_records(source_report: dict[str, Any]) -> list[dict[str, Any]]:
    categories = source_report.get("categories") or {}
    if not isinstance(categories, dict):
        return []
    records: list[dict[str, Any]] = []
    for category, rows in categories.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            feature = str(row.get("feature") or "")
            if not feature:
                continue
            item = dict(row)
            item["category"] = str(item.get("category") or category)
            records.append(item)
    return records


def _rejection_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": row.get("decision"),
        "blockers": row.get("blockers") or [],
        "deltas_vs_core": row.get("deltas_vs_core") or {},
        "metrics": row.get("metrics") or {},
        "source_report": str(SHADOW_ABLATION_PATH),
    }


def collect_rejected_ablation_features(ablation_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rejected: dict[str, dict[str, Any]] = {}
    rows = ablation_report.get("feature_ablation_results")
    if not isinstance(rows, list):
        return rejected
    for row in rows:
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature") or "")
        if not feature:
            continue
        if row.get("decision") == "reject_for_now":
            rejected[feature] = _rejection_payload(row)
    return rejected


def collect_rejected_ablation_groups(ablation_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rejected: dict[str, dict[str, Any]] = {}
    rows = ablation_report.get("group_ablation_results")
    if not isinstance(rows, list):
        return rejected
    for row in rows:
        if not isinstance(row, dict):
            continue
        group = str(row.get("feature_group") or "")
        if not group:
            continue
        if row.get("decision") == "reject_for_now":
            rejected[group] = _rejection_payload(row)
            rejected[group]["features"] = row.get("features") or []
    return rejected


def candidate_score(record: dict[str, Any]) -> float:
    group = str(record.get("feature_group") or "ungrouped")
    category = str(record.get("category") or "")
    non_zero_rate = to_float(record.get("non_zero_rate"))
    zero_rate = to_float(record.get("zero_rate"))
    missing_rate = to_float(record.get("missing_rate"))
    unique_count = to_float(record.get("unique_count"))
    base = GROUP_WEIGHTS.get(group, 2.0)
    score = base
    score += non_zero_rate * 10.0
    score += min(unique_count, 125.0) / 25.0
    score -= missing_rate * 15.0
    score -= max(0.0, zero_rate - 0.50) * 5.0
    if category == "feature_disabled_with_signal":
        score += 4.0
    if category == "feature_disabled_with_partial_signal":
        score += 2.0
    if bool(record.get("allow_in_shadow_model", False)):
        score += 2.0
    if bool(record.get("allow_in_main_model", False)):
        score -= 100.0
    return round(score, 4)


def classify_candidate(
    record: dict[str, Any],
    rejected_by_ablation: dict[str, dict[str, Any]],
    rejected_groups_by_ablation: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    feature = str(record.get("feature") or "")
    group = str(record.get("feature_group") or "ungrouped")
    category = str(record.get("category") or "")
    zero_rate = to_float(record.get("zero_rate"))
    missing_rate = to_float(record.get("missing_rate"))
    non_zero_rate = to_float(record.get("non_zero_rate"))
    allow_shadow = bool(record.get("allow_in_shadow_model", False))
    allow_main = bool(record.get("allow_in_main_model", False))

    reasons: list[str] = []
    if allow_main:
        return "already_active_core", ["feature is already allowed in active model schema"]
    if feature in rejected_by_ablation:
        details = rejected_by_ablation[feature]
        deltas = details.get("deltas_vs_core") if isinstance(details.get("deltas_vs_core"), dict) else {}
        return "blocked_shadow_ablation_rejected", [
            "feature was rejected by per-feature shadow ablation",
            "brier_delta_vs_core=" + str(deltas.get("brier_delta_vs_core")),
            "logloss_delta_vs_core=" + str(deltas.get("logloss_delta_vs_core")),
        ]
    if group in rejected_groups_by_ablation:
        details = rejected_groups_by_ablation[group]
        deltas = details.get("deltas_vs_core") if isinstance(details.get("deltas_vs_core"), dict) else {}
        return "blocked_shadow_group_ablation_rejected", [
            "feature group was rejected by group-level shadow ablation",
            "feature_group=" + group,
            "brier_delta_vs_core=" + str(deltas.get("brier_delta_vs_core")),
            "logloss_delta_vs_core=" + str(deltas.get("logloss_delta_vs_core")),
        ]
    if category in BLOCKED_CATEGORIES:
        return "blocked_source_or_review_first", ["source coverage or all-zero review must be fixed before shadow promotion"]
    if not allow_shadow:
        return "blocked_schema_not_shadow_allowed", ["feature is not allowed by current shadow schema"]
    if missing_rate > MAX_SHADOW_MISSING_RATE:
        return "blocked_missingness_too_high", ["missing rate exceeds shadow candidate threshold"]
    if non_zero_rate < MIN_SHADOW_NON_ZERO_RATE:
        return "blocked_signal_too_sparse", ["non-zero coverage is below minimum shadow threshold"]
    if category not in SHADOW_CANDIDATE_CATEGORIES:
        return "watchlist_only", ["category is not a promotion candidate category"]

    reasons.append("feature has non-zero signal and low missingness")
    reasons.append("promotion is limited to shadow evaluation; active model remains locked")
    if non_zero_rate >= MIN_DIRECT_SHADOW_NON_ZERO_RATE and zero_rate <= MAX_DIRECT_SHADOW_ZERO_RATE:
        return "shadow_candidate_now", reasons
    reasons.append("coverage is partial, so require backfill/monitoring before stronger promotion")
    return "shadow_candidate_after_backfill", reasons


def group_limited(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    for record in sorted(records, key=lambda row: (-float(row["candidate_score"]), str(row["feature"]))):
        group = str(record.get("feature_group") or "ungrouped")
        if group_counts.get(group, 0) >= MAX_CANDIDATES_PER_GROUP:
            continue
        selected.append(record)
        group_counts[group] = group_counts.get(group, 0) + 1
        if len(selected) >= MAX_TOTAL_CANDIDATES:
            break
    return selected


def build_report() -> dict[str, Any]:
    source_report = read_json(FEATURE_SOURCE_COVERAGE_PATH)
    walkforward = read_json(WALKFORWARD_PATH)
    root_cause = read_json(MODEL_ROOT_CAUSE_PATH)
    baseline = read_json(BASELINE_COMPARISON_PATH)
    ablation = read_json(SHADOW_ABLATION_PATH)
    rejected_by_ablation = collect_rejected_ablation_features(ablation)
    rejected_groups_by_ablation = collect_rejected_ablation_groups(ablation)

    records = collect_feature_records(source_report)
    evaluated: list[dict[str, Any]] = []
    for record in records:
        promotion_status, reasons = classify_candidate(record, rejected_by_ablation, rejected_groups_by_ablation)
        feature = str(record.get("feature") or "")
        group = str(record.get("feature_group") or "ungrouped")
        candidate = {
            "feature": record.get("feature"),
            "feature_group": record.get("feature_group"),
            "source_category": record.get("category"),
            "promotion_status": promotion_status,
            "candidate_score": candidate_score(record),
            "missing_rate": record.get("missing_rate"),
            "zero_rate": record.get("zero_rate"),
            "non_zero_rate": record.get("non_zero_rate"),
            "unique_count": record.get("unique_count"),
            "allow_in_main_model": bool(record.get("allow_in_main_model", False)),
            "allow_in_shadow_model": bool(record.get("allow_in_shadow_model", False)),
            "reasons": reasons,
        }
        if feature in rejected_by_ablation:
            candidate["shadow_ablation"] = rejected_by_ablation[feature]
        if group in rejected_groups_by_ablation:
            candidate["shadow_group_ablation"] = rejected_groups_by_ablation[group]
        evaluated.append(candidate)

    shadow_now = [item for item in evaluated if item["promotion_status"] == "shadow_candidate_now"]
    shadow_after_backfill = [item for item in evaluated if item["promotion_status"] == "shadow_candidate_after_backfill"]
    blocked = [item for item in evaluated if str(item["promotion_status"]).startswith("blocked")]
    active = [item for item in evaluated if item["promotion_status"] == "already_active_core"]
    shadow_ablation_rejected = [item for item in evaluated if item["promotion_status"] == "blocked_shadow_ablation_rejected"]
    shadow_group_ablation_rejected = [item for item in evaluated if item["promotion_status"] == "blocked_shadow_group_ablation_rejected"]

    recommended_shadow_set = group_limited(shadow_now + shadow_after_backfill)

    walkforward_ready = bool(walkforward.get("walkforward_ready", False))
    collapse_fold_count = int(walkforward.get("collapse_fold_count", 0) or 0)
    baseline_gate = baseline.get("quality_gate") if isinstance(baseline.get("quality_gate", None), dict) else {}
    baseline_promotion_allowed = bool(baseline_gate.get("promotion_allowed", False))

    active_promotion_blockers = []
    if not walkforward_ready:
        active_promotion_blockers.append("walkforward_sample_count_below_required_threshold")
    if collapse_fold_count > 0:
        active_promotion_blockers.append("walkforward_collapse_detected")
    if not baseline_promotion_allowed:
        active_promotion_blockers.append("baseline_quality_gate_blocks_promotion")
    if root_cause.get("status") == "blocked":
        active_promotion_blockers.append("model_data_root_cause_status_blocked")

    report = {
        "generated_at": utc_now(),
        "report_type": "feature_promotion_candidate_report",
        "status": "warning" if recommended_shadow_set else "blocked",
        "source_reports": {
            "feature_source_coverage": str(FEATURE_SOURCE_COVERAGE_PATH),
            "walkforward": str(WALKFORWARD_PATH),
            "model_data_root_cause": str(MODEL_ROOT_CAUSE_PATH),
            "baseline_comparison": str(BASELINE_COMPARISON_PATH),
            "shadow_feature_ablation": str(SHADOW_ABLATION_PATH),
        },
        "promotion_policy": {
            "active_model_promotion_allowed": False,
            "shadow_evaluation_allowed": True,
            "reason": "Candidates may enter shadow evaluation only; active model promotion requires passing walk-forward, baseline, root-cause, and ablation gates.",
            "active_promotion_blockers": sorted(set(active_promotion_blockers)),
            "minimum_shadow_non_zero_rate": MIN_SHADOW_NON_ZERO_RATE,
            "maximum_shadow_missing_rate": MAX_SHADOW_MISSING_RATE,
            "ablation_rejection_policy": "Features rejected by per-feature or group-level shadow ablation are removed from the recommended shadow set until source/data/model changes produce a new positive ablation result.",
        },
        "summary": {
            "evaluated_feature_count": len(evaluated),
            "already_active_core_count": len(active),
            "shadow_candidate_now_count": len(shadow_now),
            "shadow_candidate_after_backfill_count": len(shadow_after_backfill),
            "shadow_ablation_rejected_count": len(shadow_ablation_rejected),
            "shadow_group_ablation_rejected_count": len(shadow_group_ablation_rejected),
            "blocked_candidate_count": len(blocked),
            "recommended_shadow_set_count": len(recommended_shadow_set),
        },
        "recommended_shadow_set": recommended_shadow_set,
        "shadow_candidate_now": sorted(shadow_now, key=lambda row: (-float(row["candidate_score"]), str(row["feature"]))),
        "shadow_candidate_after_backfill": sorted(shadow_after_backfill, key=lambda row: (-float(row["candidate_score"]), str(row["feature"]))),
        "shadow_ablation_rejected": sorted(shadow_ablation_rejected, key=lambda row: str(row["feature"])),
        "shadow_group_ablation_rejected": sorted(shadow_group_ablation_rejected, key=lambda row: str(row["feature"])),
        "blocked_candidates": sorted(blocked, key=lambda row: (-float(row["candidate_score"]), str(row["feature"])))[:30],
        "already_active_core": sorted(active, key=lambda row: str(row["feature"])),
        "next_actions": [
            "Run shadow-only evaluation only for recommended_shadow_set; if empty, do not run active-model promotion work.",
            "Keep rejected shadow features and rejected feature groups quarantined until a future ablation report shows Brier/logloss improvement.",
            "Backfill source-missing all-zero features before considering new candidates.",
            "Keep active model locked until walk-forward sample size, collapse, baseline, and ablation gates pass.",
        ],
        "promotion_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def main() -> int:
    report = build_report()
    print(json.dumps({"status": report["status"], "recommended_shadow_set_count": report["summary"]["recommended_shadow_set_count"], "output_path": str(OUTPUT_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
