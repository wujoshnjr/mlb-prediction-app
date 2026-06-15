from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.feature_schema import (
        CORE_MODEL_FEATURES,
        SHADOW_CANDIDATE_FEATURES,
        TRACKING_ONLY_FEATURES,
        FEATURE_METADATA,
        MODEL_FEATURE_VERSION,
        get_model_feature_schema_hash,
    )
except ImportError:  # pragma: no cover
    from feature_schema import (  # type: ignore[no-redef]
        CORE_MODEL_FEATURES,
        SHADOW_CANDIDATE_FEATURES,
        TRACKING_ONLY_FEATURES,
        FEATURE_METADATA,
        MODEL_FEATURE_VERSION,
        get_model_feature_schema_hash,
    )

FEATURE_MISSINGNESS_PATH = Path("report/feature_missingness_report.json")
OUTPUT_PATH = Path("report/feature_priority_report.json")

GROUP_IMPACT_WEIGHTS = {
    "starting_pitcher": 9.5,
    "statcast_batting": 9.0,
    "lineup": 8.5,
    "market": 8.0,
    "bullpen": 7.5,
    "park_weather": 6.5,
    "core_strength": 6.0,
    "rest_travel": 5.5,
    "injuries": 5.0,
    "availability": 3.0,
    "ungrouped": 4.0,
    "unknown": 4.0,
}

GROUP_RATIONALES = {
    "starting_pitcher": "Starting pitcher quality, pitch traits, strikeout/walk profile and matchup context are high-leverage MLB signals.",
    "statcast_batting": "Statcast batting quality measures contact quality and batter skill signals that are closer to process than final score outcomes.",
    "lineup": "Confirmed lineup and top-order hitter context affect actual game strength and should be tracked before model promotion.",
    "market": "Market no-vig probability and odds movement are essential baselines; model promotion should not occur without beating them.",
    "bullpen": "Bullpen availability and workload affect late-game win probability, especially after starter exits.",
    "park_weather": "Park and weather context can affect run environment, totals, and matchup conditions.",
    "core_strength": "Team strength and recent form provide useful low-latency baselines but are often less granular than player-level context.",
    "rest_travel": "Rest and travel can matter, but should be lower priority than pitcher, lineup, market and Statcast process signals.",
    "injuries": "Injury context is important but must be carefully timestamped to avoid stale or post-game leakage.",
    "availability": "Availability flags are governance diagnostics, not direct model predictors by default.",
    "ungrouped": "Ungrouped feature; review manually before promotion.",
    "unknown": "Unknown group; review schema metadata before promotion.",
}

DATA_SOURCE_HINTS = {
    "starting_pitcher": [
        "confirmed starters",
        "pitcher advanced context",
        "pitch movement / Stuff+ style indicators",
        "strikeout and walk profile",
    ],
    "statcast_batting": [
        "Baseball Savant / Statcast hitter quality",
        "xwOBA or wOBA-style top lineup aggregates",
        "barrel rate / hard-hit rate / swing-miss context",
    ],
    "lineup": [
        "confirmed lineup feed",
        "projected lineup fallback",
        "top-3 hitter quality aggregation",
        "platoon split context",
    ],
    "market": [
        "bookmaker odds history",
        "no-vig implied probability",
        "opening-to-current odds movement",
        "closing-line tracking",
    ],
    "bullpen": [
        "recent bullpen innings",
        "days-rest workload",
        "high-leverage reliever availability",
    ],
    "park_weather": [
        "game-time weather",
        "wind and temperature effects",
        "park factor by venue",
    ],
    "rest_travel": [
        "schedule density",
        "travel / timezone features",
        "back-to-back and games-last-7-days context",
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def load_feature_missingness() -> dict[str, Any]:
    if not FEATURE_MISSINGNESS_PATH.exists():
        return {}
    try:
        payload = json.loads(FEATURE_MISSINGNESS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except Exception:
        return default


def feature_tier(feature: str, group: str, exists: bool, missing_rate: float, zero_rate: float | None) -> str:
    high_zero = zero_rate is not None and zero_rate >= 0.95
    if feature in CORE_MODEL_FEATURES and not exists:
        return "P0_CORE_BROKEN"
    if feature in CORE_MODEL_FEATURES and (missing_rate > 0.20 or high_zero):
        return "P0_CORE_DEGRADED"
    if group in {"starting_pitcher", "statcast_batting", "lineup", "market", "bullpen"}:
        if missing_rate >= 0.50 or high_zero or not exists:
            return "P1_HIGH_IMPACT_DATA_GAP"
        return "P1_HIGH_IMPACT_READY_FOR_RESEARCH"
    if group in {"park_weather", "core_strength", "rest_travel", "injuries"}:
        if missing_rate >= 0.50 or high_zero or not exists:
            return "P2_MEDIUM_IMPACT_DATA_GAP"
        return "P2_MEDIUM_IMPACT_READY_FOR_RESEARCH"
    if group == "availability":
        return "GOVERNANCE_FLAG_DO_NOT_PROMOTE_DIRECTLY"
    return "P3_REVIEW_MANUALLY"


def recommended_action(feature: str, tier: str, missing_rate: float, zero_rate: float | None) -> str:
    if tier.startswith("P0_CORE"):
        return "fix_before_any_model_review"
    if tier == "P1_HIGH_IMPACT_DATA_GAP":
        return "prioritize_data_source_or_feature_bridge"
    if tier == "P1_HIGH_IMPACT_READY_FOR_RESEARCH":
        return "evaluate_in_shadow_model_and_permutation_importance"
    if tier == "P2_MEDIUM_IMPACT_DATA_GAP":
        return "backfill_after_p1_sources_are_stable"
    if tier == "P2_MEDIUM_IMPACT_READY_FOR_RESEARCH":
        return "evaluate_after_p1_candidates"
    if tier == "GOVERNANCE_FLAG_DO_NOT_PROMOTE_DIRECTLY":
        return "use_as_availability_gate_not_predictor"
    if missing_rate >= 1.0:
        return "review_missing_source"
    if zero_rate is not None and zero_rate >= 0.95:
        return "review_zero_fill_or_feature_bridge"
    return "manual_review"


def priority_score(
    *,
    feature: str,
    group: str,
    exists: bool,
    missing_rate: float,
    zero_rate: float | None,
    allow_shadow: bool,
    leakage_risk: str,
) -> float:
    impact = GROUP_IMPACT_WEIGHTS.get(group, GROUP_IMPACT_WEIGHTS["unknown"])
    zero_component = zero_rate if zero_rate is not None else (1.0 if not exists else 0.0)
    gap = max(missing_rate, zero_component)
    readiness = (1.0 - min(1.0, missing_rate)) * (1.0 - min(1.0, zero_component))
    shadow_bonus = 1.0 if allow_shadow else 0.25
    leakage_penalty = 0.65 if leakage_risk == "medium" else 0.35 if leakage_risk == "high" else 1.0
    core_bonus = 1.35 if feature in CORE_MODEL_FEATURES else 1.0
    data_gap_value = impact * (0.70 * gap + 0.30 * readiness) * shadow_bonus * leakage_penalty * core_bonus
    return round(float(data_gap_value), 4)


def build_feature_priority_report() -> dict[str, Any]:
    missingness = load_feature_missingness()
    features = missingness.get("features") if isinstance(missingness.get("features"), list) else []
    recommendations: list[str] = []

    if not features:
        report = {
            "generated_at": utc_now(),
            "report_type": "feature_priority_report",
            "status": "skipped",
            "source_path": str(FEATURE_MISSINGNESS_PATH),
            "feature_schema_version": MODEL_FEATURE_VERSION,
            "feature_schema_hash": get_model_feature_schema_hash(),
            "priorities": [],
            "group_summary": {},
            "top_data_source_actions": [],
            "recommendations": [
                "Run scripts/feature_missingness_report.py before generating feature priority diagnostics."
            ],
            "promotion_allowed": False,
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
            "production_model_replacement_allowed": False,
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    priorities: list[dict[str, Any]] = []
    group_summary: dict[str, dict[str, Any]] = {}

    for raw in features:
        if not isinstance(raw, dict):
            continue
        feature = str(raw.get("feature") or "")
        if not feature:
            continue
        metadata = FEATURE_METADATA.get(feature, {})
        group = str(raw.get("feature_group") or metadata.get("group") or "unknown")
        leakage_risk = str(raw.get("leakage_risk") or metadata.get("leakage_risk") or "unknown")
        exists = bool(raw.get("exists", False))
        missing_rate = as_float(raw.get("missing_rate"), 1.0 if not exists else 0.0)
        zero_rate_raw = raw.get("zero_rate")
        zero_rate = None if zero_rate_raw is None else as_float(zero_rate_raw)
        non_zero_rate = as_float(raw.get("non_zero_rate"), 0.0)
        allow_main = bool(raw.get("allow_in_main_model", feature in CORE_MODEL_FEATURES))
        allow_shadow = bool(raw.get("allow_in_shadow_model", feature in SHADOW_CANDIDATE_FEATURES or feature in CORE_MODEL_FEATURES))
        tier = feature_tier(feature, group, exists, missing_rate, zero_rate)
        score = priority_score(
            feature=feature,
            group=group,
            exists=exists,
            missing_rate=missing_rate,
            zero_rate=zero_rate,
            allow_shadow=allow_shadow,
            leakage_risk=leakage_risk,
        )
        entry = {
            "feature": feature,
            "tier": tier,
            "priority_score": score,
            "feature_group": group,
            "group_impact_weight": GROUP_IMPACT_WEIGHTS.get(group, GROUP_IMPACT_WEIGHTS["unknown"]),
            "group_rationale": GROUP_RATIONALES.get(group, GROUP_RATIONALES["unknown"]),
            "exists": exists,
            "missing_rate": round(missing_rate, 6),
            "zero_rate": None if zero_rate is None else round(zero_rate, 6),
            "non_zero_rate": round(non_zero_rate, 6),
            "allow_in_main_model": allow_main,
            "allow_in_shadow_model": allow_shadow,
            "leakage_risk": leakage_risk,
            "recommended_action": recommended_action(feature, tier, missing_rate, zero_rate),
            "data_source_hints": DATA_SOURCE_HINTS.get(group, []),
            "promotion_note": (
                "Do not promote directly; require shadow evaluation, OOS lift, calibration and leakage review."
                if feature not in CORE_MODEL_FEATURES
                else "Core feature; maintain stability before any new model review."
            ),
        }
        priorities.append(entry)
        summary = group_summary.setdefault(
            group,
            {
                "feature_count": 0,
                "missing_or_zero_count": 0,
                "shadow_candidate_count": 0,
                "avg_priority_score": 0.0,
                "top_features": [],
            },
        )
        summary["feature_count"] += 1
        if missing_rate >= 0.50 or (zero_rate is not None and zero_rate >= 0.95) or not exists:
            summary["missing_or_zero_count"] += 1
        if allow_shadow:
            summary["shadow_candidate_count"] += 1

    priorities.sort(key=lambda item: (-float(item["priority_score"]), str(item["feature"])))

    for group, summary in group_summary.items():
        group_items = [item for item in priorities if item["feature_group"] == group]
        if group_items:
            summary["avg_priority_score"] = round(
                sum(float(item["priority_score"]) for item in group_items) / len(group_items), 4
            )
            summary["top_features"] = [item["feature"] for item in group_items[:5]]

    top_data_source_actions: list[dict[str, Any]] = []
    for group, summary in sorted(
        group_summary.items(),
        key=lambda item: (-float(item[1]["avg_priority_score"]), str(item[0])),
    ):
        if group == "availability":
            continue
        top_data_source_actions.append(
            {
                "feature_group": group,
                "avg_priority_score": summary["avg_priority_score"],
                "missing_or_zero_count": summary["missing_or_zero_count"],
                "top_features": summary["top_features"],
                "data_source_hints": DATA_SOURCE_HINTS.get(group, []),
                "rationale": GROUP_RATIONALES.get(group, GROUP_RATIONALES["unknown"]),
            }
        )

    p0 = [item["feature"] for item in priorities if str(item["tier"]).startswith("P0")]
    p1_gap = [item["feature"] for item in priorities if item["tier"] == "P1_HIGH_IMPACT_DATA_GAP"]
    p1_ready = [item["feature"] for item in priorities if item["tier"] == "P1_HIGH_IMPACT_READY_FOR_RESEARCH"]

    if p0:
        recommendations.append("Fix degraded or missing core features before model review.")
    if p1_gap:
        recommendations.append("Prioritize P1 data-source gaps before adding model complexity.")
    if p1_ready:
        recommendations.append("Evaluate P1 ready features only in shadow models with walk-forward and calibration gates.")
    recommendations.append("No tracking-only feature should enter active serving without evidence from OOS lift, baseline comparison and leakage review.")

    report = {
        "generated_at": utc_now(),
        "report_type": "feature_priority_report",
        "status": "warning" if p0 or p1_gap else "ok",
        "source_path": str(FEATURE_MISSINGNESS_PATH),
        "feature_schema_version": MODEL_FEATURE_VERSION,
        "feature_schema_hash": get_model_feature_schema_hash(),
        "feature_count": len(priorities),
        "p0_core_issue_count": len(p0),
        "p1_high_impact_data_gap_count": len(p1_gap),
        "p1_high_impact_ready_count": len(p1_ready),
        "priorities": priorities,
        "top_15_priorities": priorities[:15],
        "group_summary": group_summary,
        "top_data_source_actions": top_data_source_actions[:8],
        "recommendations": recommendations,
        "promotion_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    report = build_feature_priority_report()
    print(
        json.dumps(
            {
                "status": report["status"],
                "feature_count": report.get("feature_count", 0),
                "output_path": str(OUTPUT_PATH),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
