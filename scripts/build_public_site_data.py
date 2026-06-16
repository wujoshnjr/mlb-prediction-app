from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_DIR = Path("report")
SITE_DIR = Path("site")
SITE_DATA_DIR = SITE_DIR / "data"
OUTPUT_PATH = SITE_DATA_DIR / "public_dashboard.json"

PREDICTION_PATH = REPORT_DIR / "prediction.json"
REPORT_HEALTH_PATH = REPORT_DIR / "report_health_gate.json"
DATA_CONTRACT_PATH = REPORT_DIR / "data_contract_report.json"
REPO_ANOMALY_PATH = REPORT_DIR / "repo_anomaly_report.json"
ARTIFACT_QUARANTINE_PATH = REPORT_DIR / "artifact_quarantine_report.json"
FEATURE_PRIORITY_PATH = REPORT_DIR / "feature_priority_report.json"
BASELINE_COMPARISON_PATH = REPORT_DIR / "baseline_comparison_report.json"
MODEL_EVAL_PATH = REPORT_DIR / "model_eval_report.json"
DAILY_ACCURACY_PATH = REPORT_DIR / "daily_model_accuracy_report.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    return value if isinstance(value, str) else str(value)


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return round(parsed, digits)


def _percent(value: Any, digits: int = 1) -> float | None:
    parsed = _round(value, 6)
    if parsed is None:
        return None
    return round(parsed * 100.0, digits)


def _predictions(prediction_report: dict[str, Any]) -> list[dict[str, Any]]:
    raw = prediction_report.get("today_predictions") or prediction_report.get("predictions") or prediction_report.get("games") or []
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _status_badge(*values: Any) -> str:
    text = " ".join(str(value or "").lower() for value in values)
    if any(word in text for word in ["failed", "error", "fatal"]):
        return "danger"
    if any(word in text for word in ["blocked", "warning", "quarantined", "insufficient", "tracking"]):
        return "warning"
    if any(word in text for word in ["ok", "completed", "ready"]):
        return "ok"
    return "neutral"


def _build_game_card(item: dict[str, Any]) -> dict[str, Any]:
    governance = item.get("model_governance_status") if isinstance(item.get("model_governance_status"), dict) else {}
    data_quality = item.get("data_quality_status") if isinstance(item.get("data_quality_status"), dict) else {}
    details = item.get("recommendation_block_details") if isinstance(item.get("recommendation_block_details"), list) else []
    missing_flags = item.get("missing_signal_flags") if isinstance(item.get("missing_signal_flags"), list) else []
    context = item.get("daily_context_summary") if isinstance(item.get("daily_context_summary"), dict) else {}

    return {
        "game_id": str(item.get("game_id") or ""),
        "game_date": item.get("game_date"),
        "start_time": item.get("start_time"),
        "status": item.get("game_status") or "Preview",
        "home_team": item.get("home_team") or "Home",
        "away_team": item.get("away_team") or "Away",
        "home_win_probability_pct": _percent(item.get("predicted_home_win_pct") or item.get("displayed_home_win_pct")),
        "market_home_probability_pct": _percent(item.get("market_no_vig_home_prob")),
        "model_edge_home_pct": _percent(item.get("model_edge_home")),
        "model_disagreement_pct": _percent(item.get("model_disagreement_with_market")),
        "recommendation_label": "TRACKING ONLY",
        "recommendation_raw": item.get("recommendation") or "NO BET",
        "recommendation_status": item.get("recommendation_status") or "TRACKING_ONLY",
        "moneyline_gate_status": item.get("moneyline_gate_status"),
        "data_quality_grade": data_quality.get("data_quality_grade") or item.get("data_quality_grade"),
        "prediction_allowed": bool(data_quality.get("prediction_allowed", item.get("prediction_allowed", True))),
        "bet_allowed": False,
        "lineup_status": data_quality.get("lineup_status"),
        "pitcher_status": data_quality.get("pitcher_status"),
        "starter_confirmation_pending": bool(data_quality.get("starter_confirmation_pending", False)),
        "home_probable_pitcher_name": context.get("home_probable_pitcher_name"),
        "away_probable_pitcher_name": context.get("away_probable_pitcher_name"),
        "risk_profile": item.get("risk_profile") or "blocked",
        "risk_flags": [str(flag) for flag in missing_flags[:8]],
        "public_notes": [str(note) for note in details[:4]],
        "odds_context": {
            "status": item.get("odds_quality_status"),
            "source": item.get("odds_source"),
            "home_moneyline_decimal": _round(item.get("home_moneyline_odds"), 3),
            "away_moneyline_decimal": _round(item.get("away_moneyline_odds"), 3),
            "spread_line": _round(item.get("spread_line"), 2),
            "total_line": _round(item.get("total_line"), 2),
        },
        "governance": {
            "mode": governance.get("mode") or "paper_trading_only",
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
            "block_reasons": [str(reason) for reason in (governance.get("block_reasons") or [])[:6]],
        },
    }


def _top_feature_actions(feature_priority: dict[str, Any]) -> list[dict[str, Any]]:
    actions = feature_priority.get("top_data_source_actions") or []
    if isinstance(actions, list) and actions:
        result = []
        for item in actions[:6]:
            if not isinstance(item, dict):
                continue
            result.append(
                {
                    "feature_group": item.get("feature_group"),
                    "rationale": item.get("rationale"),
                    "top_features": [str(feature) for feature in (item.get("top_features") or [])[:5]],
                }
            )
        return result

    priorities = feature_priority.get("top_15_priorities") or []
    result = []
    if isinstance(priorities, list):
        for item in priorities[:6]:
            if not isinstance(item, dict):
                continue
            result.append(
                {
                    "feature_group": item.get("feature_group"),
                    "rationale": item.get("recommended_action"),
                    "top_features": [str(item.get("feature"))],
                }
            )
    return result


def _performance_payload(daily_accuracy: dict[str, Any]) -> dict[str, Any]:
    official = daily_accuracy.get("official_accuracy") if isinstance(daily_accuracy.get("official_accuracy"), dict) else {}
    rolling = daily_accuracy.get("rolling_windows") if isinstance(daily_accuracy.get("rolling_windows"), dict) else {}
    slices = daily_accuracy.get("slices") if isinstance(daily_accuracy.get("slices"), dict) else {}
    pending = daily_accuracy.get("pending_predictions") if isinstance(daily_accuracy.get("pending_predictions"), dict) else {}
    clv = daily_accuracy.get("clv_metrics") if isinstance(daily_accuracy.get("clv_metrics"), dict) else {}
    daily = daily_accuracy.get("daily_accuracy") if isinstance(daily_accuracy.get("daily_accuracy"), list) else []
    latest_daily = next((item for item in reversed(daily) if isinstance(item, dict) and item.get("sample_count") not in (None, 0)), {})
    return {
        "official_accuracy": official,
        "rolling_windows": rolling,
        "slices": slices,
        "pending_predictions": pending,
        "clv_metrics": clv,
        "latest_daily": latest_daily,
        "interpretation": daily_accuracy.get("interpretation") if isinstance(daily_accuracy.get("interpretation"), dict) else {},
    }


def build_public_dashboard() -> dict[str, Any]:
    prediction = _read_json(PREDICTION_PATH)
    report_health = _read_json(REPORT_HEALTH_PATH)
    data_contract = _read_json(DATA_CONTRACT_PATH)
    repo_anomaly = _read_json(REPO_ANOMALY_PATH)
    artifact_quarantine = _read_json(ARTIFACT_QUARANTINE_PATH)
    feature_priority = _read_json(FEATURE_PRIORITY_PATH)
    baseline = _read_json(BASELINE_COMPARISON_PATH)
    model_eval = _read_json(MODEL_EVAL_PATH)
    daily_accuracy = _read_json(DAILY_ACCURACY_PATH)

    games = [_build_game_card(item) for item in _predictions(prediction)]
    governance = prediction.get("model_governance") if isinstance(prediction.get("model_governance"), dict) else {}
    quality_gate = baseline.get("quality_gate") if isinstance(baseline.get("quality_gate"), dict) else {}
    model_metrics = model_eval.get("metrics") if isinstance(model_eval.get("metrics"), dict) else {}
    performance = _performance_payload(daily_accuracy)
    official_accuracy = performance.get("official_accuracy") if isinstance(performance.get("official_accuracy"), dict) else {}
    clv_metrics = performance.get("clv_metrics") if isinstance(performance.get("clv_metrics"), dict) else {}

    dashboard = {
        "generated_at": _utc_now(),
        "source_generated_at": prediction.get("generated_at"),
        "pipeline_version": prediction.get("pipeline_version"),
        "site_version": "public_dashboard_v2_paper_board",
        "public_disclaimer": "Research dashboard only. No betting advice, no automated wagering, and no live-betting enablement.",
        "hero": {
            "title": "MLB Paper Prediction Board",
            "subtitle": "每日 MLB 賽事重點整理、paper-only 訊號、已結算戰績與模型治理狀態。",
            "primary_status": "Research mode · tracking only",
            "status_badge": "warning",
        },
        "system_status": {
            "pipeline_health_status": report_health.get("pipeline_health_status") or report_health.get("status"),
            "model_quality_status": report_health.get("model_quality_status"),
            "data_contract_status": data_contract.get("status"),
            "repo_anomaly_status": repo_anomaly.get("status"),
            "artifact_status": artifact_quarantine.get("status"),
            "promotion_allowed": False,
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
            "production_model_replacement_allowed": False,
            "badge": _status_badge(report_health.get("pipeline_health_status"), report_health.get("model_quality_status"), repo_anomaly.get("status")),
        },
        "metrics": {
            "scheduled_game_count": prediction.get("scheduled_game_count"),
            "game_count": len(games),
            "clean_settled_sample_count": official_accuracy.get("sample_count") or ((governance.get("model_governance") or {}).get("clean_settled_sample_count") if isinstance(governance.get("model_governance"), dict) else governance.get("clean_model_sample_count")),
            "minimum_train_samples": governance.get("min_clean_train_samples"),
            "avg_clv": _round(clv_metrics.get("avg_clv", governance.get("avg_clv")), 4),
            "positive_clv_rate_pct": _percent(clv_metrics.get("positive_clv_rate", governance.get("positive_clv_rate"))),
            "repo_anomaly_errors": repo_anomaly.get("error_count"),
            "data_contract_errors": data_contract.get("error_count"),
            "model_auc": _round(model_metrics.get("roc_auc"), 4),
            "model_brier": _round(model_metrics.get("brier"), 4),
            "official_accuracy_pct": _percent(official_accuracy.get("accuracy")),
            "daily_accuracy_status": daily_accuracy.get("status"),
        },
        "performance": performance,
        "governance_summary": {
            "block_reasons": [str(reason) for reason in (governance.get("block_reasons") or [])[:8]],
            "model_quality_blocks": [str(item) for item in (report_health.get("model_quality_blocks") or [])[:10]],
            "data_contract_warnings": [str(item) for item in (data_contract.get("warnings") or [])[:8]],
            "repo_anomaly_warnings": repo_anomaly.get("warning_count"),
            "baseline_gate_status": quality_gate.get("status"),
            "baseline_gate_reasons": [str(reason) for reason in (quality_gate.get("reasons") or [])[:8]],
            "artifact_quarantine": {
                "status": artifact_quarantine.get("status"),
                "stale_sample_mismatch": artifact_quarantine.get("stale_sample_mismatch"),
                "artifact_training_sample_count": artifact_quarantine.get("artifact_training_sample_count"),
                "current_training_sample_count": artifact_quarantine.get("current_training_sample_count"),
                "minimum_production_training_samples": artifact_quarantine.get("minimum_production_training_samples"),
            },
        },
        "feature_roadmap": {
            "status": feature_priority.get("status"),
            "feature_count": feature_priority.get("feature_count"),
            "p0_core_issue_count": feature_priority.get("p0_core_issue_count"),
            "p1_high_impact_data_gap_count": feature_priority.get("p1_high_impact_data_gap_count"),
            "actions": _top_feature_actions(feature_priority),
        },
        "games": games,
        "navigation": [
            {"label": "Today", "href": "#games"},
            {"label": "Record", "href": "#record"},
            {"label": "Governance", "href": "#governance"},
            {"label": "Roadmap", "href": "#roadmap"},
            {"label": "About", "href": "#about"},
        ],
    }
    return dashboard


def main() -> int:
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dashboard = build_public_dashboard()
    OUTPUT_PATH.write_text(json.dumps(_json_safe(dashboard), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_path": str(OUTPUT_PATH), "game_count": len(dashboard.get("games", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
