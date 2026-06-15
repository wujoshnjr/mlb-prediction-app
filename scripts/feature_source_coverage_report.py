from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_DIR = Path("report")
FEATURE_MISSINGNESS_PATH = REPORT_DIR / "feature_missingness_report.json"
OUTPUT_PATH = REPORT_DIR / "feature_source_coverage_report.json"

ALL_ZERO_THRESHOLD = 0.95
HIGH_ZERO_THRESHOLD = 0.50
HIGH_MISSING_THRESHOLD = 0.50
PROMOTION_SIGNAL_NON_ZERO_THRESHOLD = 0.35
LOW_MISSING_THRESHOLD = 0.10

GROUP_PRIORITY = {
    "starting_pitcher": 10,
    "statcast_batting": 9,
    "lineup": 9,
    "market": 8,
    "bullpen": 8,
    "park_weather": 6,
    "core_strength": 6,
    "rest_travel": 5,
    "injuries": 5,
    "availability": 4,
    "ungrouped": 3,
}

LIKELY_STRUCTURAL_ZERO_FEATURES = {
    "is_day_game",
    "rest_diff",
    "back2back_diff",
    "games_last_3d_diff",
    "games_last_7d_diff",
    "rest_pressure_diff",
    "wind_effect",
    "temp_effect",
    "precip_effect",
    "timezone_diff",
}

SOURCE_BACKFILL_FEATURES = {
    "pitch_movement_diff",
    "pitch_type_matchup_score",
    "platoon_ops_diff",
    "top3_woba_diff",
    "statcast_woba_diff",
    "avg_bat_speed_diff",
    "barrel_pa_diff",
    "hardhit_pa_diff",
    "swing_miss_diff",
    "odds_change",
    "injury_diff",
    "catcher_era_diff",
    "cs_diff",
    "bullpen_availability_diff",
    "bullpen_ip_diff",
    "sp_fip_diff",
    "sp_csw_diff",
    "sp_stuff_plus_diff",
    "k_pct_diff",
    "bb_pct_diff",
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


def classify_feature(item: dict[str, Any]) -> tuple[str, list[str]]:
    feature = str(item.get("feature") or "")
    group = str(item.get("feature_group") or "ungrouped")
    zero_rate = to_float(item.get("zero_rate"))
    missing_rate = to_float(item.get("missing_rate"))
    non_zero_rate = to_float(item.get("non_zero_rate"))
    allow_main = bool(item.get("allow_in_main_model", False))
    allow_shadow = bool(item.get("allow_in_shadow_model", False))

    reasons: list[str] = []

    if allow_main and missing_rate < LOW_MISSING_THRESHOLD and zero_rate < HIGH_ZERO_THRESHOLD:
        reasons.append("active model feature with acceptable coverage")
        return "active_core_healthy", reasons

    if missing_rate >= HIGH_MISSING_THRESHOLD:
        reasons.append("historical coverage missing for at least half of samples")
        if allow_shadow and not allow_main:
            reasons.append("feature is tracking-only; do not promote until backfilled")
        return "source_missing_or_not_backfilled", reasons

    if zero_rate >= ALL_ZERO_THRESHOLD and non_zero_rate == 0:
        if feature in LIKELY_STRUCTURAL_ZERO_FEATURES:
            reasons.append("all-zero but feature can be structurally zero in many games")
            return "true_zero_or_schedule_sparse", reasons
        if feature in SOURCE_BACKFILL_FEATURES or group in {"starting_pitcher", "statcast_batting", "lineup", "market", "injuries", "bullpen"}:
            reasons.append("high-impact source appears absent or not backfilled")
            if allow_shadow and not allow_main:
                reasons.append("feature is tracking-only because evidence is insufficient")
            return "source_missing_all_zero", reasons
        reasons.append("all-zero feature needs manual source review")
        return "all_zero_needs_review", reasons

    if zero_rate >= HIGH_ZERO_THRESHOLD:
        if feature in LIKELY_STRUCTURAL_ZERO_FEATURES or group in {"rest_travel", "park_weather"}:
            reasons.append("high-zero context feature may be naturally sparse")
            return "true_zero_or_context_sparse", reasons
        reasons.append("more than half of samples are zero; source or transform may be incomplete")
        if allow_shadow and not allow_main:
            reasons.append("tracking-only feature has partial signal but is not ready for active model")
            return "feature_disabled_with_partial_signal", reasons
        return "high_zero_needs_review", reasons

    if allow_shadow and not allow_main and non_zero_rate >= PROMOTION_SIGNAL_NON_ZERO_THRESHOLD and missing_rate < LOW_MISSING_THRESHOLD:
        reasons.append("tracking-only feature has non-zero coverage and may be a shadow candidate")
        return "feature_disabled_with_signal", reasons

    if not allow_main and not allow_shadow:
        reasons.append("feature is excluded from active and shadow model schemas")
        return "excluded_by_schema", reasons

    reasons.append("coverage acceptable for tracking, not active model")
    return "tracking_coverage_ok", reasons


def priority_score(item: dict[str, Any], category: str) -> float:
    feature = str(item.get("feature") or "")
    group = str(item.get("feature_group") or "ungrouped")
    zero_rate = to_float(item.get("zero_rate"))
    missing_rate = to_float(item.get("missing_rate"))
    non_zero_rate = to_float(item.get("non_zero_rate"))
    base = GROUP_PRIORITY.get(group, 3)
    score = float(base)
    score += missing_rate * 6.0
    score += zero_rate * 4.0
    if category in {"source_missing_all_zero", "source_missing_or_not_backfilled"}:
        score += 5.0
    if category in {"feature_disabled_with_signal", "feature_disabled_with_partial_signal"}:
        score += non_zero_rate * 5.0
    if feature in SOURCE_BACKFILL_FEATURES:
        score += 3.0
    return round(score, 4)


def build_report() -> dict[str, Any]:
    missingness = read_json(FEATURE_MISSINGNESS_PATH)
    features = missingness.get("features") or []
    if not isinstance(features, list):
        features = []

    categories: dict[str, list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []

    for item in features:
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature") or "")
        if not feature:
            continue
        category, reasons = classify_feature(item)
        record = {
            "feature": feature,
            "feature_group": item.get("feature_group"),
            "category": category,
            "priority_score": priority_score(item, category),
            "exists": bool(item.get("exists", False)),
            "missing_rate": round(to_float(item.get("missing_rate")), 4),
            "zero_rate": round(to_float(item.get("zero_rate")), 4),
            "non_zero_rate": round(to_float(item.get("non_zero_rate")), 4),
            "unique_count": item.get("unique_count"),
            "allow_in_main_model": bool(item.get("allow_in_main_model", False)),
            "allow_in_shadow_model": bool(item.get("allow_in_shadow_model", False)),
            "recommended_action": item.get("recommended_action"),
            "reasons": reasons,
        }
        records.append(record)
        categories.setdefault(category, []).append(record)

    for bucket in categories.values():
        bucket.sort(key=lambda row: (-float(row["priority_score"]), str(row["feature"])))

    root_summary = {
        "feature_count": len(records),
        "active_core_healthy_count": len(categories.get("active_core_healthy", [])),
        "source_missing_all_zero_count": len(categories.get("source_missing_all_zero", [])),
        "source_missing_or_not_backfilled_count": len(categories.get("source_missing_or_not_backfilled", [])),
        "feature_disabled_with_signal_count": len(categories.get("feature_disabled_with_signal", [])),
        "feature_disabled_with_partial_signal_count": len(categories.get("feature_disabled_with_partial_signal", [])),
        "true_zero_or_context_sparse_count": len(categories.get("true_zero_or_context_sparse", [])) + len(categories.get("true_zero_or_schedule_sparse", [])),
        "excluded_by_schema_count": len(categories.get("excluded_by_schema", [])),
    }

    priority_actions = [
        {
            "rank": 1,
            "action": "Backfill market movement",
            "target_features": ["odds_change", "odds_available", "market_no_vig_home_prob"],
            "why": "Market movement and no-vig probability are required comparison baselines; missing movement makes edge quality harder to audit.",
        },
        {
            "rank": 2,
            "action": "Backfill lineup and top-order hitting context",
            "target_features": ["lineup_context_available", "top3_woba_diff", "platoon_ops_diff", "catcher_era_diff", "cs_diff"],
            "why": "Lineup quality materially changes team strength and should not be silently represented as zero.",
        },
        {
            "rank": 3,
            "action": "Backfill starting pitcher advanced context",
            "target_features": ["sp_fip_diff", "sp_csw_diff", "sp_stuff_plus_diff", "pitch_movement_diff", "pitch_type_matchup_score"],
            "why": "Starting pitcher features are high-leverage MLB predictors and are currently sparse or all-zero.",
        },
        {
            "rank": 4,
            "action": "Backfill bullpen availability",
            "target_features": ["bullpen_availability_diff", "bullpen_ip_diff", "bullpen_context_available"],
            "why": "Bullpen workload and availability affect late-game win probability and need source-level freshness.",
        },
        {
            "rank": 5,
            "action": "Keep missing flags separate from numeric zeros",
            "target_features": ["*_available", "*_missing_indicator"],
            "why": "Do not let source_missing become numeric zero; preserve missingness as explicit model information.",
        },
    ]

    report = {
        "generated_at": utc_now(),
        "report_type": "feature_source_coverage_report",
        "status": "warning" if root_summary["source_missing_all_zero_count"] or root_summary["source_missing_or_not_backfilled_count"] else "ok",
        "source_report": str(FEATURE_MISSINGNESS_PATH),
        "root_summary": root_summary,
        "categories": {key: value[:25] for key, value in sorted(categories.items())},
        "top_priority_features": sorted(records, key=lambda row: (-float(row["priority_score"]), str(row["feature"])))[:25],
        "priority_actions": priority_actions,
        "interpretation": {
            "source_missing": "The upstream data source or historical backfill is absent; do not treat the zero as a real baseball value.",
            "true_zero_or_context_sparse": "The zero may be valid because the condition often does not occur, but it still needs freshness checks.",
            "feature_disabled_with_signal": "The feature has some non-zero coverage but remains excluded from active model training until governance promotion evidence exists.",
            "active_core_healthy": "The feature is currently allowed in the active model and has acceptable coverage.",
        },
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
    print(
        json.dumps(
            {
                "status": report["status"],
                "feature_count": report["root_summary"]["feature_count"],
                "source_missing_all_zero_count": report["root_summary"]["source_missing_all_zero_count"],
                "source_missing_or_not_backfilled_count": report["root_summary"]["source_missing_or_not_backfilled_count"],
                "output_path": str(OUTPUT_PATH),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
