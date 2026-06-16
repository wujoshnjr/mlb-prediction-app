from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPORT_DIR = Path("report")
OUTPUT_PATH = REPORT_DIR / "pipeline_manifest.json"

TRACKED_FILES = [
    "report/prediction.json",
    "report/evaluation_clv_diagnostic.json",
    "report/market_edge_research.json",
    "report/feature_promotion_report.json",
    "report/feature_availability_diagnostic.json",
    "report/feature_zero_root_cause_diagnostic.json",
    "report/feature_grade_report.json",
    "report/feature_priority_report.json",
    "report/feature_source_coverage_report.json",
    "report/feature_promotion_candidate_report.json",
    "report/shadow_feature_experiment_report.json",
    "report/shadow_feature_experiment_rows.csv",
    "report/shadow_feature_decision_report.json",
    "report/shadow_feature_ablation_report.json",
    "report/shadow_feature_ablation_rows.csv",
    "report/model_data_root_cause_report.json",
    "report/baseline_comparison_report.json",
    "report/clv_by_edge_bucket.json",
    "report/clv_by_side.json",
    "report/clv_by_odds_range.json",
    "report/clv_by_lineup_status.json",
    "report/calibration_report.json",
    "report/per_slice_performance_report.json",
    "report/walkforward_evaluation.json",
    "report/rolling_walkforward_evaluation.json",
    "report/rolling_walkforward_predictions.csv",
    "report/lineup_starter_slice_report.json",
    "report/market_close_report.json",
    "report/research_quality_report.json",
    "report/settle_reliability_report.json",
    "report/settled_prediction_link_report.json",
    "report/finalized_linkage_diagnostic_report.json",
    "report/snapshot_sanitization_report.json",
    "report/prediction_sanitization_report.json",
    "report/model_registry_report.json",
    "report/promotion_gate_report.json",
    "report/decision_audit_report.json",
    "report/decision_audit.csv",
    "report/paper_trading_ledger_report.json",
    "report/risk_exposure_report.json",
    "report/artifact_retention_manifest.json",
    "report/world_class_trading_system_report.json",
    "report/saas_readiness_report.json",
    "report/model_artifact_status_report.json",
    "report/model_status_consistency_report.json",
    "report/artifact_quarantine_report.json",
    "report/repo_anomaly_report.json",
    "report/report_health_gate.json",
    "report/artifact_rebuild_readiness_report.json",
    "report/feature_contract_report.json",
    "report/feature_missingness_report.json",
    "report/model_eval_report.json",
    "report/prediction_collapse_report.json",
    "report/train_ensemble_report.json",
    "report/sample_state_report.json",
    "report/training_samples_report.json",
    "report/outcome_linkage_diagnostic.json",
    "report/data_contract_report.json",
    "report/pipeline_manifest.json",
    "report/index.html",
    "report/walkforward_predictions.csv",
    "report/walk_forward_validation_report.json",
    "report/calibration_diagnostics_report.json",
    "report/prediction_trust_report.json",
    "report/model_comparison_report.json",
    "report/model_decision_guardrail_report.json",
    "report/shadow_ensemble_stack_report.json",
    "report/research_promotion_readiness_report.json",
    "report/underdog_diagnostic_report.json",
    "report/confidence_bucket_guardrail_report.json",
    "report/slice_promotion_gate_report.json",
    "report/feature_freshness_report.json",
    "report/lineup_quality_report.json",
    "report/model_correctness_report.json",
    "report/daily_model_accuracy_report.json",
    "report/accuracy_root_cause_report.json",
    "report/away_pick_diagnostic_report.json",
    "report/away_guardrail_impact_report.json",
    "report/odds_fetch_diagnostic.json",
    "report/odds_matching_diagnostic.json",
    "report/edge_sanity_guardrail_report.json",
    "report/signal_quality_report.json",
    "report/product_experience_report.json",
    "site/index.html",
    "site/styles.css",
    "site/app.js",
    "site/data/public_dashboard.json",
    "scripts/accuracy_root_cause_report.py",
    "scripts/build_public_site_data.py",
    "scripts/model_data_root_cause_report.py",
    "scripts/feature_source_coverage_report.py",
    "scripts/feature_promotion_candidate_report.py",
    "scripts/shadow_feature_experiment_report.py",
    "scripts/shadow_feature_decision_report.py",
    "scripts/shadow_feature_ablation_report.py",
    "wrangler.toml",
    "data/lineup_quality_context.csv",
    "data/finalized_snapshot_outcomes.csv",
    "data/walk_forward_predictions.csv",
    "data/model_lab/shadow_ensemble_stack.pkl",
    "data/model_registry.json",
    "data/model_lab/logistic_baseline.pkl",
    "data/model_lab/lightgbm_classifier.pkl",
    "data/model_lab/xgboost_classifier.pkl",
    "data/model_lab/lightgbm_market_residual.pkl",
    "data/sample_state.json",
    "data/training_samples.csv",
    "data/model_artifact_status.json",
    "data/oos_predictions_with_labels.csv",
    "data/paper_trading_ledger.csv",
    "data/training_status.json",
    "data/prediction_snapshots.csv",
    "data/market_odds_history.csv",
    "data/finalized_games.csv",
    "data/daily_game_context.csv",
    "data/weather_context.csv",
    "data/top3_player_context.csv",
    "data/savant_top3_context.csv",
    "data/pitcher_advanced_context.csv",
    "data/team_form_context.csv",
    "data/context_feature_bridge.csv",
    "data/projected_lineup_context.csv",
    "README.md",
    "docs/SAAS_ROADMAP.md",
    "docs/API_DESIGN.md",
    "docs/RISK_POLICY.md",
    "docs/DATA_SOURCES.md",
    "docs/EVALUATION_METHOD.md",
    "docs/DEPLOYMENT_RENDER.md",
    "docs/GITHUB_ACTIONS_WORKFLOW.md",
    "docs/NO_AUTOMATED_WAGERING_POLICY.md",
    "docs/B2B_PRODUCT_SPEC.md",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_summary(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"json_valid": False, "json_error": str(exc)}
    if not isinstance(data, dict):
        return {"json_valid": True, "json_type": type(data).__name__}
    summary: Dict[str, Any] = {"json_valid": True, "json_type": "dict"}
    for key in (
        "generated_at",
        "timestamp",
        "status",
        "report_type",
        "error_count",
        "warning_count",
        "research_grade",
        "risk_status",
        "feature_schema_hash",
        "site_version",
    ):
        if key in data:
            summary[key] = data.get(key)
    predictions = data.get("predictions") or data.get("today_predictions") or data.get("games")
    if isinstance(predictions, list):
        summary["prediction_count"] = len(predictions)
    slices = data.get("slices")
    if isinstance(slices, list):
        summary["slice_count"] = len(slices)
    elif isinstance(slices, dict):
        summary["slice_group_count"] = len(slices)
    bins = data.get("bins") or data.get("reliability_table")
    if isinstance(bins, list):
        summary["bin_count"] = len(bins)
    rows = data.get("rows") or data.get("features") or data.get("priorities") or data.get("issues") or data.get("games")
    if isinstance(rows, list):
        summary["row_count_in_json"] = len(rows)
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        for key in ("game_count", "repo_anomaly_errors", "data_contract_errors", "model_auc", "model_brier"):
            if key in metrics:
                summary[key] = metrics.get(key)
    for key in (
        "settled_prediction_count",
        "total_oos_predictions",
        "audit_count",
        "ledger_count",
        "file_count",
        "tracked_file_count",
        "missing_file_count",
        "scanned_file_count",
        "model_quality_block_count",
        "core_feature_count",
        "feature_count",
        "p0_core_issue_count",
        "p1_high_impact_data_gap_count",
        "sample_count",
        "valid_sample_count",
        "promotion_allowed",
        "production_allowed",
        "live_betting_allowed",
        "automated_wagering_allowed",
        "production_model_replacement_allowed",
        "quarantined",
        "stale_sample_mismatch",
    ):
        if key in data:
            summary[key] = data.get(key)
    return summary


def _csv_row_count(path: Path) -> Optional[int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            rows = sum(1 for _ in reader)
    except Exception:
        return None
    return max(0, rows - 1)


def _file_record(path_text: str) -> Dict[str, Any]:
    path = Path(path_text)
    exists = path.exists()
    size_bytes = path.stat().st_size if exists and path.is_file() else 0
    record: Dict[str, Any] = {
        "path": path_text,
        "exists": exists,
        "size_bytes": size_bytes,
        "sha256": _sha256(path),
    }
    if path.suffix.lower() == ".json" and exists:
        record.update(_json_summary(path))
    if path.suffix.lower() == ".csv" and exists:
        record["row_count"] = _csv_row_count(path)
    return record


def build_manifest() -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = [_file_record(path) for path in TRACKED_FILES]
    missing_files = [item["path"] for item in files if not item["exists"]]
    invalid_json_files = [
        item["path"]
        for item in files
        if item["path"].endswith(".json") and item["exists"] and item.get("json_valid") is False
    ]
    status = "ok"
    if invalid_json_files:
        status = "failed"
    elif missing_files:
        status = "partial"
    recommendations = []
    if missing_files:
        recommendations.append(
            "Some tracked files are missing; check whether they are optional, not generated yet, or failed upstream."
        )
    if invalid_json_files:
        recommendations.append(
            "Some tracked JSON files are invalid; fix these before trusting generated reports."
        )
    if not recommendations:
        recommendations.append("All tracked pipeline artifacts are present and readable.")
    report = {
        "generated_at": _utc_now(),
        "status": status,
        "tracked_file_count": len(files),
        "missing_file_count": len(missing_files),
        "invalid_json_file_count": len(invalid_json_files),
        "missing_files": missing_files,
        "invalid_json_files": invalid_json_files,
        "files": files,
        "recommendations": recommendations,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    manifest = build_manifest()
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "tracked_file_count": manifest["tracked_file_count"],
                "missing_file_count": manifest["missing_file_count"],
                "invalid_json_file_count": manifest["invalid_json_file_count"],
                "output_path": str(OUTPUT_PATH),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
