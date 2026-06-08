from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REPORT_DIR = Path("report")

OUTPUT_HTML = REPORT_DIR / "index.html"

SOURCES = {
    "prediction": REPORT_DIR / "prediction.json",
    "baseline_comparison": REPORT_DIR / "baseline_comparison_report.json",
    "clv_edge": REPORT_DIR / "clv_by_edge_bucket.json",
    "clv_side": REPORT_DIR / "clv_by_side.json",
    "clv_odds": REPORT_DIR / "clv_by_odds_range.json",
    "clv_lineup": REPORT_DIR / "clv_by_lineup_status.json",
    "calibration": REPORT_DIR / "calibration_report.json",
    "walkforward": REPORT_DIR / "walkforward_evaluation.json",
    "rolling_walkforward": REPORT_DIR / "rolling_walkforward_evaluation.json",
    "lineup_starter_slice": REPORT_DIR / "lineup_starter_slice_report.json",
    "market_close": REPORT_DIR / "market_close_report.json",
    "research_quality": REPORT_DIR / "research_quality_report.json",
    "settle_reliability": REPORT_DIR / "settle_reliability_report.json",
    "settled_prediction_link": REPORT_DIR / "settled_prediction_link_report.json",
    "finalized_linkage_diagnostic": REPORT_DIR / "finalized_linkage_diagnostic_report.json",
    "model_registry": REPORT_DIR / "model_registry_report.json",
    "promotion_gate": REPORT_DIR / "promotion_gate_report.json",
    "decision_audit": REPORT_DIR / "decision_audit_report.json",
    "paper_trading_ledger": REPORT_DIR / "paper_trading_ledger_report.json",
    "risk_exposure": REPORT_DIR / "risk_exposure_report.json",
    "artifact_retention": REPORT_DIR / "artifact_retention_manifest.json",
    "world_class_trading_system": REPORT_DIR / "world_class_trading_system_report.json",
    "saas_readiness": REPORT_DIR / "saas_readiness_report.json",
    "sample_state": Path("data/sample_state.json"),
    "sample_state_report": REPORT_DIR / "sample_state_report.json",
    "data_contract": REPORT_DIR / "data_contract_report.json",
    "pipeline_manifest": REPORT_DIR / "pipeline_manifest.json",
    "feature_availability": REPORT_DIR / "feature_availability_diagnostic.json",
    "feature_zero": REPORT_DIR / "feature_zero_root_cause_diagnostic.json",
    "feature_grade": REPORT_DIR / "feature_grade_report.json",
    "feature_promotion": REPORT_DIR / "feature_promotion_report.json",
    "training_status": Path("data/training_status.json"),
    "walk_forward_validation": REPORT_DIR / "walk_forward_validation_report.json",
    "calibration_diagnostics": REPORT_DIR / "calibration_diagnostics_report.json",
    "prediction_trust": REPORT_DIR / "prediction_trust_report.json",
    "model_comparison": REPORT_DIR / "model_comparison_report.json",
    "shadow_ensemble_stack": REPORT_DIR / "shadow_ensemble_stack_report.json",
    "research_promotion_readiness": REPORT_DIR / "research_promotion_readiness_report.json",
}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _escape(value: Any) -> str:
    if value is None:
        return "unavailable"
    return html.escape(str(value))


def _predictions(report: Optional[Dict[str, Any]]) -> list[dict[str, Any]]:
    if not report:
        return []
    raw = report.get("predictions") or report.get("today_predictions") or report.get("games") or []
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _first_prediction(report: Optional[Dict[str, Any]]) -> dict[str, Any]:
    preds = _predictions(report)
    return preds[0] if preds else {}


def _section(title: str, body: str) -> str:
    return f"<section class='section'><h2>{_escape(title)}</h2>{body}</section>"


def _list_items(items: list[Any]) -> str:
    if not items:
        return "<p class='muted'>None</p>"
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ul>"


def _baseline_table(report: Optional[Dict[str, Any]]) -> str:
    if not report:
        return "<p class='muted'>Baseline report unavailable.</p>"

    baselines = report.get("baselines") or {}
    if not isinstance(baselines, dict) or not baselines:
        return "<p class='muted'>No baseline metrics available.</p>"

    rows = [
        "<table><tr><th>Baseline</th><th>Count</th><th>Brier</th><th>Logloss</th><th>Accuracy</th></tr>"
    ]
    for name, metrics in baselines.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{_escape(name)}</td>"
            f"<td>{_escape(metrics.get('count'))}</td>"
            f"<td>{_escape(metrics.get('brier'))}</td>"
            f"<td>{_escape(metrics.get('logloss'))}</td>"
            f"<td>{_escape(metrics.get('accuracy'))}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _clv_table(title: str, report: Optional[Dict[str, Any]]) -> str:
    if not report:
        return f"<h3>{_escape(title)}</h3><p class='muted'>Unavailable.</p>"

    slices = report.get("slices") or []
    if not slices:
        return f"<h3>{_escape(title)}</h3><p class='muted'>No slices available.</p>"

    rows = [
        f"<h3>{_escape(title)}</h3>",
        "<table><tr><th>Slice</th><th>Count</th><th>Avg CLV</th><th>Positive CLV Rate</th><th>Block Live</th></tr>",
    ]
    for item in slices:
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{_escape(item.get('slice'))}</td>"
            f"<td>{_escape(item.get('count'))}</td>"
            f"<td>{_escape(item.get('avg_clv'))}</td>"
            f"<td>{_escape(item.get('positive_clv_rate'))}</td>"
            f"<td>{_escape(item.get('block_live_bet'))}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _calibration_table(report: Optional[Dict[str, Any]]) -> str:
    if not report:
        return "<p class='muted'>Calibration report unavailable.</p>"

    body = (
        f"<p>Total count: {_escape(report.get('total_count'))}</p>"
        f"<p>Calibration ready: {_escape(report.get('calibration_ready'))}</p>"
        f"<p>Weighted ECE: {_escape(report.get('weighted_ece'))}</p>"
    )

    bins = report.get("bins") or []
    if not bins:
        return body

    rows = [
        "<table><tr><th>Bin</th><th>Count</th><th>Avg Predicted</th><th>Actual Win Rate</th><th>Error</th></tr>"
    ]
    for item in bins:
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{_escape(item.get('bin'))}</td>"
            f"<td>{_escape(item.get('count'))}</td>"
            f"<td>{_escape(item.get('avg_predicted'))}</td>"
            f"<td>{_escape(item.get('actual_win_rate'))}</td>"
            f"<td>{_escape(item.get('calibration_error'))}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return body + "\n" + "\n".join(rows)


def build_html() -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = {key: _read_json(path) for key, path in SOURCES.items()}

    prediction = data["prediction"]
    first = _first_prediction(prediction)
    governance = first.get("model_governance_status") if isinstance(first, dict) else {}
    if not isinstance(governance, dict):
        governance = {}

    data_quality = first.get("data_quality_status") if isinstance(first, dict) else {}
    if not isinstance(data_quality, dict):
        data_quality = {}

    training = data["training_status"] or {}
    sample_state = data["sample_state"] or data["sample_state_report"] or {}
    baseline = data["baseline_comparison"]
    calibration = data["calibration"]
    walkforward = data["walkforward"]
    feature_availability = data["feature_availability"] or {}
    feature_zero = data["feature_zero"] or {}
    feature_grade = data["feature_grade"] or {}
    feature_promotion = data["feature_promotion"] or {}

    html_parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>MLB Prediction App Dashboard</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:2rem;background:#f7f7f7;color:#222;}",
        ".banner{background:#fff3cd;border:1px solid #ffe08a;padding:1rem;border-radius:8px;}",
        ".section{background:#fff;padding:1rem;margin:1rem 0;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08);}",
        ".muted{color:#666;font-style:italic;}",
        "table{border-collapse:collapse;width:100%;margin:.75rem 0;}th,td{border:1px solid #ddd;padding:.5rem;text-align:left;}th{background:#eee;}",
        "code{background:#eee;padding:.1rem .25rem;border-radius:4px;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>MLB Prediction App Dashboard</h1>",
        "<div class='banner'>"
        "<strong>Current status:</strong> experimental paper-trading only.<br>"
        "<strong>Live betting:</strong> disabled by governance.<br>"
        "For research and educational use only. No guaranteed profit. Not financial or gambling advice."
        "</div>",
    ]

    html_parts.append(
        _section(
            "Model Readiness",
            "<p>Train-eligible samples: "
            + _escape(sample_state.get("train_eligible_samples", training.get("sample_count")))
            + "</p><p>Clean settled snapshots: "
            + _escape(sample_state.get("clean_settled_snapshots"))
            + "</p><p>Minimum clean train samples: "
            + _escape(sample_state.get("minimum_clean_train_samples", training.get("minimum_clean_train_samples")))
            + "</p><p>Training allowed: "
            + _escape(sample_state.get("training_allowed"))
            + "</p><p>Promotion sample ready: "
            + _escape(sample_state.get("promotion_sample_ready"))
            + "</p><p>Walk-forward predictions: "
            + _escape(sample_state.get("walkforward_predictions"))
            + "</p><p>Model trained: "
            + _escape(sample_state.get("trained", training.get("trained")))
            + "</p><p>Model type: "
            + _escape(training.get("model_type"))
            + "</p>",
        )
    )

    html_parts.append(
        _section(
            "Governance",
            "<p>Mode: "
            + _escape(governance.get("mode"))
            + "</p><p>Live betting allowed: "
            + _escape(governance.get("live_betting_allowed"))
            + "</p><p>Shadow live allowed: "
            + _escape(governance.get("shadow_live_allowed"))
            + "</p><h3>Block reasons</h3>"
            + _list_items(governance.get("block_reasons") or []),
        )
    )

    html_parts.append(
        _section(
            "Data Quality",
            "<p>Grade: "
            + _escape(data_quality.get("data_quality_grade"))
            + "</p><p>Prediction allowed: "
            + _escape(data_quality.get("prediction_allowed"))
            + "</p><p>Bet allowed: "
            + _escape(data_quality.get("bet_allowed"))
            + "</p><h3>Missing critical sources</h3>"
            + _list_items(data_quality.get("missing_critical_sources") or [])
            + "<h3>Missing important sources</h3>"
            + _list_items(data_quality.get("missing_important_sources") or []),
        )
    )

    html_parts.append(_section("Baseline Comparison", _baseline_table(baseline)))

    clv_body = "\n".join(
        [
            _clv_table("Edge Bucket", data["clv_edge"]),
            _clv_table("Side", data["clv_side"]),
            _clv_table("Odds Range", data["clv_odds"]),
            _clv_table("Lineup Status", data["clv_lineup"]),
        ]
    )
    html_parts.append(_section("CLV Summary", clv_body))

    html_parts.append(_section("Calibration Summary", _calibration_table(calibration)))

    html_parts.append(
        _section(
            "Walk-forward Status",
            "<p>Status: "
            + _escape(walkforward.get("status") if walkforward else "unavailable")
            + "</p><p>Ready: "
            + _escape(walkforward.get("walkforward_ready") if walkforward else False)
            + "</p><p>Total OOS predictions: "
            + _escape(walkforward.get("total_oos_predictions") if walkforward else None)
            + "</p>",
        )
    )

    html_parts.append(
        _section(
            "Feature Health",
            "<p>Availability high risk features: "
            + _escape(feature_availability.get("high_risk_features"))
            + "</p><p>Zero root cause still-zero features: "
            + _escape(feature_zero.get("still_zero_features"))
            + "</p><p>Feature grade counts: "
            + _escape(feature_grade.get("grade_counts"))
            + "</p><p>Feature promotion candidates: "
            + _escape(feature_promotion.get("candidate_shadow_count"))
            + "</p><p>Ready for review: "
            + _escape(feature_promotion.get("ready_for_review_count"))
            + "</p>",
        )
    )

    html_parts.append(
        _section(
            "Risk Disclosure",
            "<p>This project is for research and educational use only. It does not provide financial advice, gambling advice, guaranteed profit, or betting instructions. Sports betting is risky and markets can be highly efficient. Paper trading is the default and live betting is disabled by governance.</p>",
        )
    )

    # Engineering grade summary
    engineering_rows = [
        "<table><tr><th>Report</th><th>Status</th><th>Key Metric</th><th>Notes</th></tr>"
    ]

    engineering_sources = {
        "Settle Reliability": data.get("settle_reliability"),
        "Settled Prediction Link": data.get("settled_prediction_link"),
        "Rolling Walk-forward": data.get("rolling_walkforward"),
        "Feature Promotion": data.get("feature_promotion"),
        "Lineup / Starter Slice": data.get("lineup_starter_slice"),
        "Market Close": data.get("market_close"),
        "Research Quality": data.get("research_quality"),
        "Model Registry": data.get("model_registry"),
        "Promotion Gate": data.get("promotion_gate"),
        "Decision Audit": data.get("decision_audit"),
        "Paper Trading Ledger": data.get("paper_trading_ledger"),
        "Risk Exposure": data.get("risk_exposure"),
        "Artifact Retention": data.get("artifact_retention"),
        "World-Class Trading System": data.get("world_class_trading_system"),
        "SaaS Readiness": data.get("saas_readiness"),
        "Sample State": data.get("sample_state") or data.get("sample_state_report"),
        "Data Contract": data.get("data_contract"),
        "Pipeline Manifest": data.get("pipeline_manifest"),
        "Walk-forward Validation": data.get("walk_forward_validation"),
        "Calibration Diagnostics": data.get("calibration_diagnostics"),
        "Prediction Trust": data.get("prediction_trust"),
        "Model Comparison": data.get("model_comparison"),
        "Shadow Ensemble Stack": data.get("shadow_ensemble_stack"),
        "Finalized Linkage": data.get("finalized_linkage_diagnostic"),
        "Research Promotion Readiness": data.get("research_promotion_readiness"),
    }

    for title, report_data in engineering_sources.items():
        if not report_data:
            engineering_rows.append(
                f"<tr><td>{_escape(title)}</td><td>unavailable</td><td>-</td><td>Report missing</td></tr>"
            )
            continue

        status = report_data.get("status", report_data.get("risk_status", "unavailable"))

        key_metric = "-"
        if title == "Settle Reliability":
            key_metric = f"settle_rate={_escape(report_data.get('settle_rate'))}"
        elif title == "Settled Prediction Link":
            key_metric = (
                f"linked_games={_escape(report_data.get('linked_game_count'))}; "
                f"link_rate={_escape(report_data.get('link_rate'))}"
            )
        elif title == "Rolling Walk-forward":
            key_metric = f"oos={_escape(report_data.get('total_oos_predictions'))}"
        elif title == "Feature Promotion":
            key_metric = (
                f"candidates={_escape(report_data.get('candidate_shadow_count'))}; "
                f"ready={_escape(report_data.get('ready_for_review_count'))}; "
                f"sample={_escape(report_data.get('sample_count'))}"
            )
        elif title == "Research Quality":
            key_metric = f"grade={_escape(report_data.get('research_grade'))}"
        elif title == "Promotion Gate":
            key_metric = f"promotion_allowed={_escape(report_data.get('promotion_allowed'))}"
        elif title == "Decision Audit":
            key_metric = f"audit_count={_escape(report_data.get('audit_count'))}"
        elif title == "Paper Trading Ledger":
            key_metric = f"ledger_count={_escape(report_data.get('ledger_count'))}"
        elif title == "Risk Exposure":
            key_metric = f"open_units={_escape(report_data.get('total_open_paper_units'))}"
        elif title == "World-Class Trading System":
            key_metric = (
                f"score={_escape(report_data.get('overall_score'))}; "
                f"stage={_escape(report_data.get('world_class_stage'))}; "
                f"grade={_escape(report_data.get('overall_grade'))}"
            )
        elif title == "SaaS Readiness":
            key_metric = (
                f"stage={_escape(report_data.get('current_product_stage'))}; "
                f"docs={_escape(report_data.get('documentation_score'))}; "
                f"governance={_escape(report_data.get('governance_score'))}; "
                f"b2b_api_ready={_escape(report_data.get('b2b_api_ready'))}"
            )
        elif title == "Sample State":
            key_metric = (
                f"train_eligible={_escape(report_data.get('train_eligible_samples'))}; "
                f"clean_settled={_escape(report_data.get('clean_settled_snapshots'))}; "
                f"walkforward={_escape(report_data.get('walkforward_predictions'))}"
            )
        elif title == "Pipeline Manifest":
            key_metric = f"tracked={_escape(report_data.get('tracked_file_count'))}"
        elif title == "Walk-forward Validation":
            key_metric = (
                f"oos={_escape(report_data.get('total_oos_predictions'))}; "
                f"ready={_escape(report_data.get('walkforward_ready'))}"
            )
        elif title == "Calibration Diagnostics":
            key_metric = (
                f"sample={_escape(report_data.get('sample_count'))}; "
                f"ready={_escape(report_data.get('calibration_ready'))}; "
                f"shrinkage={_escape(report_data.get('recommended_probability_shrinkage'))}"
            )
        elif title == "Prediction Trust":
            key_metric = (
                f"predictions={_escape(report_data.get('prediction_count'))}; "
                f"trust={_escape(report_data.get('trust_counts'))}"
            )
        elif title == "Model Comparison":
            key_metric = (
                f"champion={_escape(report_data.get('recommended_champion'))}; "
                f"challenger={_escape(report_data.get('recommended_challenger'))}"
            )
        elif title == "Shadow Ensemble Stack":
            key_metric = (
                f"sample={_escape(report_data.get('sample_count'))}; "
                f"recommended={_escape(report_data.get('recommended_shadow_ensemble'))}; "
                f"promotion={_escape(report_data.get('promotion_eligible'))}"
            )
        elif title == "Finalized Linkage":
            key_metric = (
                f"overlap={_escape(report_data.get('overlap_count_after'))}; "
                f"written={_escape(report_data.get('api_final_written_count'))}; "
                f"pending={_escape(report_data.get('pending_not_final_count'))}; "
                f"failed={_escape(report_data.get('api_not_found_or_failed_count'))}"
            )
        elif title == "Research Promotion Readiness":
            key_metric = (
                f"status={_escape(report_data.get('status'))}; "
                f"score={_escape(report_data.get('readiness_score'))}; "
                f"allowed={_escape(report_data.get('research_promotion_allowed'))}; "
                f"challenger={_escape(report_data.get('recommended_challenger'))}"
            )

        recommendations = report_data.get("recommendations", [])
        if isinstance(recommendations, list) and recommendations:
            notes = recommendations[0]
        else:
            notes = ""

        engineering_rows.append(
            f"<tr><td>{_escape(title)}</td><td>{_escape(status)}</td><td>{_escape(key_metric)}</td><td>{_escape(notes)}</td></tr>"
        )

    engineering_rows.append("</table>")

    html_parts.append(
        _section(
            "Engineering Grade Summary",
            "\n".join(engineering_rows),
        )
    )

    html_parts.extend(
        [
            f"<footer><p class='muted'>Generated at {_escape(datetime.now(timezone.utc).isoformat())}</p></footer>",
            "</body>",
            "</html>",
        ]
    )

    html_content = "\n".join(html_parts)
    OUTPUT_HTML.write_text(html_content, encoding="utf-8")
    return html_content


def main() -> None:
    build_html()
    print(json.dumps({"status": "ok", "output_path": str(OUTPUT_HTML)}, indent=2))


if __name__ == "__main__":
    main()
