from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_DIR = Path("report")
DATA_DIR = Path("data")
OUTPUT_PATH = REPORT_DIR / "feature_source_freshness_audit.json"

TRAINING_SAMPLES_PATH = DATA_DIR / "training_samples.csv"
FEATURE_SOURCE_COVERAGE_PATH = REPORT_DIR / "feature_source_coverage_report.json"
SHADOW_ABLATION_PATH = REPORT_DIR / "shadow_feature_ablation_report.json"
ACCURACY_ROOT_CAUSE_PATH = REPORT_DIR / "accuracy_root_cause_report.json"

SOURCE_HINTS: dict[str, dict[str, Any]] = {
    "k_pct_diff": {"group": "starting_pitcher", "sources": ["data/pitcher_advanced_context.csv"], "columns": ["k_pct_diff", "home_sp_k_pct", "away_sp_k_pct"]},
    "bb_pct_diff": {"group": "starting_pitcher", "sources": ["data/pitcher_advanced_context.csv"], "columns": ["bb_pct_diff", "home_sp_bb_pct", "away_sp_bb_pct"]},
    "sp_csw_diff": {"group": "starting_pitcher", "sources": ["data/pitcher_advanced_context.csv"], "columns": ["sp_csw_diff", "home_sp_csw_proxy", "away_sp_csw_proxy"]},
    "sp_fip_diff": {"group": "starting_pitcher", "sources": ["data/pitcher_advanced_context.csv"], "columns": ["sp_fip_diff", "home_sp_fip", "away_sp_fip"]},
    "sp_stuff_plus_diff": {"group": "starting_pitcher", "sources": ["data/pitcher_advanced_context.csv"], "columns": ["sp_stuff_plus_diff", "home_sp_stuff_plus_proxy", "away_sp_stuff_plus_proxy"]},
    "avg_bat_speed_diff": {"group": "statcast_batting", "sources": ["data/savant_top3_context.csv"], "columns": ["avg_bat_speed_diff", "top3_avg_launch_speed_diff", "home_top3_avg_launch_speed", "away_top3_avg_launch_speed"]},
    "barrel_pa_diff": {"group": "statcast_batting", "sources": ["data/savant_top3_context.csv"], "columns": ["barrel_pa_diff", "top3_barrel_rate_diff", "home_top3_barrel_rate", "away_top3_barrel_rate"]},
    "hardhit_pa_diff": {"group": "statcast_batting", "sources": ["data/savant_top3_context.csv"], "columns": ["hardhit_pa_diff", "top3_hard_hit_rate_diff", "home_top3_hard_hit_rate", "away_top3_hard_hit_rate"]},
    "statcast_woba_diff": {"group": "statcast_batting", "sources": ["data/savant_top3_context.csv"], "columns": ["statcast_woba_diff", "top3_xwoba_diff", "home_top3_xwoba", "away_top3_xwoba"]},
    "top3_woba_diff": {"group": "lineup", "sources": ["data/savant_top3_context.csv", "data/top3_player_context.csv"], "columns": ["top3_woba_diff", "home_top3_woba", "away_top3_woba"]},
    "bullpen_ip_diff": {"group": "bullpen", "sources": ["data/daily_game_context.csv", "data/context_feature_bridge.csv"], "columns": ["bullpen_ip_diff", "home_bullpen_ip", "away_bullpen_ip"]},
    "bullpen_availability_diff": {"group": "bullpen", "sources": ["data/daily_game_context.csv", "data/context_feature_bridge.csv"], "columns": ["bullpen_availability_diff", "home_bullpen_availability", "away_bullpen_availability"]},
    "lag30_winrate_diff": {"group": "rest_travel", "sources": ["data/team_form_context.csv", "data/context_feature_bridge.csv"], "columns": ["lag30_winrate_diff", "home_lag30_winrate", "away_lag30_winrate"]},
}

DATE_COLUMNS = ("game_date", "date", "snapshot_date", "captured_date")
DATETIME_COLUMNS = ("captured_at", "created_at", "updated_at", "prediction_generated_at", "start_time")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    return str(value)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        parsed = float(text)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _round(value: Any, digits: int = 4) -> float | None:
    parsed = _num(value)
    return None if parsed is None else round(parsed, digits)


def _load_csv(path: Path, limit: int | None = None) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for idx, row in enumerate(reader):
                if limit is not None and idx >= limit:
                    break
                rows.append(dict(row))
            return rows, list(reader.fieldnames or [])
    except Exception:
        return [], []


def _column_stats(rows: list[dict[str, str]], column: str) -> dict[str, Any]:
    total = len(rows)
    if total <= 0:
        return {"exists": False, "row_count": 0, "non_null_count": 0, "missing_rate": None, "zero_rate": None, "non_zero_rate": None, "unique_count": 0}
    if not rows or column not in rows[0]:
        return {"exists": False, "row_count": total, "non_null_count": 0, "missing_rate": 1.0, "zero_rate": None, "non_zero_rate": None, "unique_count": 0}
    values = [row.get(column) for row in rows]
    non_null_values = [value for value in values if value is not None and str(value).strip() != ""]
    numeric_values = [_num(value) for value in non_null_values]
    numeric_values = [value for value in numeric_values if value is not None]
    zero_count = sum(1 for value in numeric_values if abs(value) < 1e-12)
    non_null_count = len(non_null_values)
    unique_count = len(set(str(value).strip() for value in non_null_values))
    return {
        "exists": True,
        "row_count": total,
        "non_null_count": non_null_count,
        "missing_rate": round(1.0 - (non_null_count / total), 4),
        "zero_rate": round(zero_count / len(numeric_values), 4) if numeric_values else None,
        "non_zero_rate": round(1.0 - (zero_count / len(numeric_values)), 4) if numeric_values else None,
        "unique_count": unique_count,
        "numeric_count": len(numeric_values),
    }


def _date_range(rows: list[dict[str, str]], fields: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"date_column": None, "min_date": None, "max_date": None, "captured_column": None, "max_captured_at": None}
    for column in DATE_COLUMNS:
        if column not in fields:
            continue
        values = sorted({str(row.get(column, "")).strip()[:10] for row in rows if str(row.get(column, "")).strip()})
        if values:
            result["date_column"] = column
            result["min_date"] = values[0]
            result["max_date"] = values[-1]
            break
    for column in DATETIME_COLUMNS:
        if column not in fields:
            continue
        values = sorted({str(row.get(column, "")).strip() for row in rows if str(row.get(column, "")).strip()})
        if values:
            result["captured_column"] = column
            result["max_captured_at"] = values[-1]
            break
    return result


def _game_ids(rows: list[dict[str, str]]) -> set[str]:
    result = set()
    for row in rows:
        value = row.get("game_id")
        if value is None:
            continue
        text = str(value).strip()
        if text:
            result.add(text)
    return result


def _training_feature_candidates(feature_source_coverage: dict[str, Any], shadow_ablation: dict[str, Any], accuracy_root_cause: dict[str, Any]) -> list[str]:
    features: list[str] = []
    for payload in (shadow_ablation.get("feature_ablation_results"),):
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and item.get("feature"):
                    features.append(str(item["feature"]))
    categories = feature_source_coverage.get("categories") if isinstance(feature_source_coverage.get("categories"), dict) else {}
    for category_name in ("feature_disabled_with_partial_signal", "source_missing_all_zero", "source_missing_or_not_backfilled", "all_zero_needs_review"):
        items = categories.get(category_name)
        if isinstance(items, list):
            for item in items[:20]:
                if isinstance(item, dict) and item.get("feature"):
                    features.append(str(item["feature"]))
    examples = accuracy_root_cause.get("feature_gap_examples") if isinstance(accuracy_root_cause.get("feature_gap_examples"), list) else []
    for item in examples:
        if isinstance(item, dict) and item.get("feature"):
            features.append(str(item["feature"]))
    ordered = []
    seen = set()
    for feature in features:
        if feature in seen:
            continue
        seen.add(feature)
        ordered.append(feature)
    return ordered


def _source_audit_for_feature(feature: str, hint: dict[str, Any], training_rows: list[dict[str, str]], training_game_ids: set[str]) -> dict[str, Any]:
    training_stats = _column_stats(training_rows, feature)
    source_results = []
    for source_path_text in hint.get("sources", []):
        source_path = Path(source_path_text)
        rows, fields = _load_csv(source_path)
        source_game_ids = _game_ids(rows)
        overlap = len(source_game_ids & training_game_ids) if source_game_ids and training_game_ids else 0
        column_results = []
        for column in hint.get("columns", []):
            stats = _column_stats(rows, column)
            if stats.get("exists"):
                column_results.append({"column": column, **stats})
        date_info = _date_range(rows, fields)
        source_results.append(
            {
                "path": source_path_text,
                "exists": source_path.exists(),
                "row_count": len(rows),
                "column_count": len(fields),
                "date_range": date_info,
                "unique_game_id_count": len(source_game_ids),
                "training_game_id_overlap_count": overlap,
                "training_game_id_overlap_rate": round(overlap / len(training_game_ids), 4) if training_game_ids else None,
                "expected_columns": hint.get("columns", []),
                "found_expected_columns": [item["column"] for item in column_results],
                "column_stats": column_results,
            }
        )
    issue_flags: list[str] = []
    if not training_stats.get("exists"):
        issue_flags.append("feature_missing_from_training_samples")
    elif (training_stats.get("zero_rate") or 0) >= 0.5:
        issue_flags.append("training_feature_zero_rate_over_50pct")
    if source_results and all(not item.get("exists") for item in source_results):
        issue_flags.append("all_hint_source_files_missing")
    if source_results and all((item.get("training_game_id_overlap_rate") or 0) < 0.4 for item in source_results if item.get("exists")):
        issue_flags.append("low_game_id_overlap_with_training_samples")
    if source_results and all(len(item.get("found_expected_columns") or []) == 0 for item in source_results if item.get("exists")):
        issue_flags.append("expected_source_columns_not_found")
    recommendation = "ok_to_continue_monitoring"
    if "feature_missing_from_training_samples" in issue_flags:
        recommendation = "fix_training_sample_feature_generation"
    elif "all_hint_source_files_missing" in issue_flags:
        recommendation = "connect_or_generate_source_file"
    elif "expected_source_columns_not_found" in issue_flags:
        recommendation = "fix_source_column_mapping"
    elif "low_game_id_overlap_with_training_samples" in issue_flags:
        recommendation = "backfill_source_history_to_training_game_ids"
    elif "training_feature_zero_rate_over_50pct" in issue_flags:
        recommendation = "audit_zero_fill_and_missing_value_policy"

    return {
        "feature": feature,
        "feature_group": hint.get("group", "unknown"),
        "training_stats": training_stats,
        "source_results": source_results,
        "issue_flags": issue_flags,
        "recommendation": recommendation,
        "promotion_allowed": False,
    }


def build_report() -> dict[str, Any]:
    feature_source_coverage = _read_json(FEATURE_SOURCE_COVERAGE_PATH)
    shadow_ablation = _read_json(SHADOW_ABLATION_PATH)
    accuracy_root_cause = _read_json(ACCURACY_ROOT_CAUSE_PATH)
    training_rows, training_fields = _load_csv(TRAINING_SAMPLES_PATH)
    training_game_ids = _game_ids(training_rows)
    candidate_features = _training_feature_candidates(feature_source_coverage, shadow_ablation, accuracy_root_cause)
    # Always include the domain-critical features that currently explain the accuracy root causes.
    for feature in SOURCE_HINTS:
        if feature not in candidate_features:
            candidate_features.append(feature)

    audits = []
    unmapped_features = []
    for feature in candidate_features:
        hint = SOURCE_HINTS.get(feature)
        if not hint:
            unmapped_features.append(feature)
            continue
        audits.append(_source_audit_for_feature(feature, hint, training_rows, training_game_ids))

    recommendation_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    for item in audits:
        recommendation_counts[item["recommendation"]] = recommendation_counts.get(item["recommendation"], 0) + 1
        for flag in item.get("issue_flags") or []:
            issue_counts[flag] = issue_counts.get(flag, 0) + 1

    highest_priority = [
        item
        for item in audits
        if item.get("recommendation")
        in {
            "fix_training_sample_feature_generation",
            "connect_or_generate_source_file",
            "fix_source_column_mapping",
            "backfill_source_history_to_training_game_ids",
            "audit_zero_fill_and_missing_value_policy",
        }
    ]
    highest_priority = sorted(highest_priority, key=lambda item: (len(item.get("issue_flags") or []), item.get("feature", "")), reverse=True)[:10]

    report = {
        "generated_at": _utc_now(),
        "report_type": "feature_source_freshness_audit",
        "status": "warning" if highest_priority else "ok",
        "summary": {
            "training_samples_available": TRAINING_SAMPLES_PATH.exists(),
            "training_sample_count": len(training_rows),
            "training_feature_count": len(training_fields),
            "training_unique_game_id_count": len(training_game_ids),
            "candidate_feature_count": len(candidate_features),
            "audited_feature_count": len(audits),
            "unmapped_feature_count": len(unmapped_features),
            "highest_priority_issue_count": len(highest_priority),
        },
        "recommendation_counts": recommendation_counts,
        "issue_counts": issue_counts,
        "highest_priority": [
            {
                "feature": item.get("feature"),
                "feature_group": item.get("feature_group"),
                "recommendation": item.get("recommendation"),
                "issue_flags": item.get("issue_flags"),
                "training_zero_rate": (item.get("training_stats") or {}).get("zero_rate"),
                "best_source_overlap_rate": max(
                    [(source.get("training_game_id_overlap_rate") or 0) for source in item.get("source_results", [])],
                    default=None,
                ),
            }
            for item in highest_priority
        ],
        "audits": audits,
        "unmapped_features": unmapped_features,
        "corrective_actions": [
            "Backfill source files so their game_id coverage overlaps training_samples before promoting features.",
            "Replace silent zero-fills with explicit missing indicators when source values are unavailable.",
            "For statcast and pitcher advanced features, verify source column mapping before shadow promotion.",
            "Keep all audited features out of the active model until Brier and logloss improve in time-ordered validation.",
        ],
        "promotion_allowed": False,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }
    return report


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUTPUT_PATH.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report.get("status"), "output_path": str(OUTPUT_PATH), "audited_feature_count": report.get("summary", {}).get("audited_feature_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
