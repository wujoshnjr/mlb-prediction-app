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
    "model_eval": REPORT_DIR / "model_eval_report.json",
    "prediction_collapse": REPORT_DIR / "prediction_collapse_report.json",
    "baseline_comparison": REPORT_DIR / "baseline_comparison_report.json",
    "walkforward": REPORT_DIR / "walkforward_evaluation.json",
    "calibration": REPORT_DIR / "calibration_report.json",
    "feature_missingness": REPORT_DIR / "feature_missingness_report.json",
    "feature_priority": REPORT_DIR / "feature_priority_report.json",
    "artifact_quarantine": REPORT_DIR / "artifact_quarantine_report.json",
    "data_contract": REPORT_DIR / "data_contract_report.json",
    "report_health_gate": REPORT_DIR / "report_health_gate.json",
    "pipeline_manifest": REPORT_DIR / "pipeline_manifest.json",
    "training_status": Path("data/training_status.json"),
    "sample_state": Path("data/sample_state.json"),
    "promotion_gate": REPORT_DIR / "promotion_gate_report.json",
    "product_experience": REPORT_DIR / "product_experience_report.json",
    "risk_exposure": REPORT_DIR / "risk_exposure_report.json",
    "daily_model_accuracy": REPORT_DIR / "daily_model_accuracy_report.json",
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


def _status_class(value: Any) -> str:
    status = str(value or "unknown").strip().lower()
    if status in {"ok", "passed", "completed", "ready"}:
        return "ok"
    if status in {"warning", "blocked", "quarantined", "needs_review", "insufficient_samples", "skipped", "partial"}:
        return "warn"
    if status in {"failed", "error", "fatal"}:
        return "bad"
    return "neutral"


def _badge(label: str, status: Any = "neutral") -> str:
    return f"<span class='badge {_status_class(status)}'>{_escape(label)}</span>"


def _value(value: Any, fallback: str = "--") -> str:
    if value is None or value == "":
        return fallback
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return _escape(value)


def _metric_card(label: str, value: Any, caption: str, status: Any = "neutral") -> str:
    return (
        f"<article class='metric-card {_status_class(status)}'>"
        f"<div class='metric-label'>{_escape(label)}</div>"
        f"<div class='metric-value'>{_value(value)}</div>"
        f"<div class='metric-caption'>{_escape(caption)}</div>"
        "</article>"
    )


def _list_items(items: Any, empty: str = "None") -> str:
    if not isinstance(items, list) or not items:
        return f"<p class='muted'>{_escape(empty)}</p>"
    return "<ul class='clean-list'>" + "".join(f"<li>{_escape(item)}</li>" for item in items[:12]) + "</ul>"


def _details(title: str, body: str, open_by_default: bool = False) -> str:
    opened = " open" if open_by_default else ""
    return f"<details class='details'{opened}><summary>{_escape(title)}</summary>{body}</details>"


def _predictions(report: Optional[Dict[str, Any]]) -> list[dict[str, Any]]:
    if not report:
        return []
    raw = report.get("predictions") or report.get("today_predictions") or report.get("games") or []
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _first_prediction(report: Optional[Dict[str, Any]]) -> dict[str, Any]:
    preds = _predictions(report)
    return preds[0] if preds else {}


def _extract_governance(prediction_report: Optional[Dict[str, Any]]) -> dict[str, Any]:
    first = _first_prediction(prediction_report)
    governance = first.get("model_governance_status") if isinstance(first, dict) else {}
    return governance if isinstance(governance, dict) else {}


def _extract_data_quality(prediction_report: Optional[Dict[str, Any]]) -> dict[str, Any]:
    first = _first_prediction(prediction_report)
    quality = first.get("data_quality_status") if isinstance(first, dict) else {}
    return quality if isinstance(quality, dict) else {}


def _topline_status(data: dict[str, Optional[Dict[str, Any]]]) -> tuple[str, str, list[str]]:
    prediction_count = len(_predictions(data.get("prediction")))
    model_eval = data.get("model_eval") or {}
    collapse = data.get("prediction_collapse") or {}
    baseline = data.get("baseline_comparison") or {}
    artifact = data.get("artifact_quarantine") or {}
    health = data.get("report_health_gate") or {}

    reasons: list[str] = []
    if prediction_count == 0:
        reasons.append("no predictions available")
    if collapse.get("do_not_promote") is True or collapse.get("model_has_no_discrimination_power") is True:
        reasons.append("model collapse guardrail active")
    if baseline.get("promotion_allowed") is False:
        reasons.append("model has not cleared baseline promotion gate")
    quality_gate = baseline.get("quality_gate") if isinstance(baseline.get("quality_gate"), dict) else {}
    if quality_gate.get("status") == "blocked":
        reasons.extend([str(item) for item in (quality_gate.get("reasons") or [])[:4]])
    if artifact.get("quarantined") is True or str(artifact.get("status", "")).lower() == "quarantined":
        reasons.append("stale artifact quarantined")
    if health.get("pipeline_health_status") == "failed":
        reasons.append("pipeline health gate has errors")

    if health.get("pipeline_health_status") == "failed":
        return "Pipeline needs review", "bad", reasons
    if reasons:
        return "Research mode · promotion locked", "warn", reasons
    if model_eval.get("status") == "ok":
        return "Research dashboard healthy", "ok", ["reports are readable and model remains paper-only"]
    return "Waiting for full diagnostics", "neutral", reasons or ["some diagnostic reports are not available yet"]


def _baseline_table(report: Optional[Dict[str, Any]]) -> str:
    if not report:
        return "<p class='muted'>Baseline report unavailable.</p>"
    baselines = report.get("baselines") or {}
    if not isinstance(baselines, dict) or not baselines:
        return "<p class='muted'>No baseline metrics available.</p>"
    rows = ["<div class='table-wrap'><table><tr><th>Baseline</th><th>Count</th><th>Brier</th><th>Logloss</th><th>Accuracy</th></tr>"]
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
    rows.append("</table></div>")
    return "\n".join(rows)


def _model_quality_section(data: dict[str, Optional[Dict[str, Any]]]) -> str:
    model_eval = data.get("model_eval") or {}
    metrics = model_eval.get("metrics") if isinstance(model_eval.get("metrics"), dict) else {}
    collapse = model_eval.get("collapse_guardrail") if isinstance(model_eval.get("collapse_guardrail"), dict) else {}
    standalone_collapse = data.get("prediction_collapse") or {}
    baseline = data.get("baseline_comparison") or {}
    walkforward = data.get("walkforward") or {}

    cards = "".join(
        [
            _metric_card("AUC", metrics.get("roc_auc"), "discrimination; 0.50 is random", "warn" if metrics.get("roc_auc") and float(metrics.get("roc_auc")) < 0.53 else "neutral"),
            _metric_card("Brier", metrics.get("brier"), "lower is better; 0.25 is random-ish", "warn"),
            _metric_card("Logloss", metrics.get("logloss"), "lower is better; 0.693 is 50/50", "warn"),
            _metric_card("Collapse", collapse.get("status", standalone_collapse.get("status")), "prediction collapse guardrail", collapse.get("status", standalone_collapse.get("status"))),
            _metric_card("Baseline Gate", (baseline.get("quality_gate") or {}).get("status"), "must beat 50%, history, and market", (baseline.get("quality_gate") or {}).get("status")),
            _metric_card("Walk-forward", walkforward.get("status"), f"folds={walkforward.get('fold_count', 'n/a')} collapse={walkforward.get('collapse_fold_count', 'n/a')}", walkforward.get("status")),
        ]
    )
    detail = "".join(
        [
            _details("Collapse reasons", _list_items(collapse.get("reasons") or standalone_collapse.get("collapse_reasons") or [])),
            _details("Baseline comparison", _baseline_table(baseline)),
            _details("Walk-forward fold summary", _walkforward_table(walkforward)),
        ]
    )
    return f"<section class='panel'><div class='section-title'><h2>Model Quality</h2><span>Probability, baseline and collapse gates</span></div><div class='metric-grid'>{cards}</div>{detail}</section>"


def _walkforward_table(report: Optional[Dict[str, Any]]) -> str:
    if not report:
        return "<p class='muted'>Walk-forward report unavailable.</p>"
    folds = report.get("folds") or []
    if not isinstance(folds, list) or not folds:
        return "<p class='muted'>No fold summaries available.</p>"
    rows = ["<div class='table-wrap'><table><tr><th>Fold</th><th>Samples</th><th>Brier</th><th>Logloss</th><th>Collapse</th><th>Reasons</th></tr>"]
    for fold in folds[:12]:
        if not isinstance(fold, dict):
            continue
        model = fold.get("model") if isinstance(fold.get("model"), dict) else {}
        reasons = fold.get("collapse_reasons") if isinstance(fold.get("collapse_reasons"), list) else []
        rows.append(
            "<tr>"
            f"<td>{_escape(fold.get('fold_id'))}</td>"
            f"<td>{_escape(fold.get('sample_count'))}</td>"
            f"<td>{_escape(model.get('brier'))}</td>"
            f"<td>{_escape(model.get('logloss'))}</td>"
            f"<td>{_escape(fold.get('collapse_detected'))}</td>"
            f"<td>{_escape(', '.join(map(str, reasons[:4])))}</td>"
            "</tr>"
        )
    rows.append("</table></div>")
    return "\n".join(rows)


def _feature_priority_section(report: Optional[Dict[str, Any]]) -> str:
    if not report:
        body = "<p class='muted'>Feature priority report unavailable.</p>"
        return f"<section class='panel'><div class='section-title'><h2>Feature Roadmap</h2><span>Data-source priority</span></div>{body}</section>"
    cards = "".join(
        [
            _metric_card("Features", report.get("feature_count"), "tracked in priority report", report.get("status")),
            _metric_card("P1 Data Gaps", report.get("p1_high_impact_data_gap_count"), "highest impact missing/zero gaps", "warn" if report.get("p1_high_impact_data_gap_count") else "ok"),
            _metric_card("P1 Ready", report.get("p1_high_impact_ready_count"), "shadow research candidates", "neutral"),
            _metric_card("Promotion", report.get("promotion_allowed"), "always locked by this diagnostic", "warn"),
        ]
    )
    priorities = report.get("top_15_priorities") or []
    rows = ["<div class='table-wrap'><table><tr><th>Feature</th><th>Tier</th><th>Group</th><th>Score</th><th>Action</th></tr>"]
    for item in priorities[:15]:
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{_escape(item.get('feature'))}</td>"
            f"<td>{_escape(item.get('tier'))}</td>"
            f"<td>{_escape(item.get('feature_group'))}</td>"
            f"<td>{_escape(item.get('priority_score'))}</td>"
            f"<td>{_escape(item.get('recommended_action'))}</td>"
            "</tr>"
        )
    rows.append("</table></div>")
    actions = report.get("top_data_source_actions") or []
    action_cards = "".join(
        f"<article class='action-card'><strong>{_escape(item.get('feature_group'))}</strong><p>{_escape(item.get('rationale'))}</p><small>{_escape(', '.join(item.get('top_features') or []))}</small></article>"
        for item in actions[:6]
        if isinstance(item, dict)
    )
    return f"<section class='panel'><div class='section-title'><h2>Feature Roadmap</h2><span>Which data source to fix first</span></div><div class='metric-grid'>{cards}</div><div class='action-grid'>{action_cards}</div>{''.join(rows)}</section>"


def _governance_section(data: dict[str, Optional[Dict[str, Any]]]) -> str:
    prediction = data.get("prediction")
    governance = _extract_governance(prediction)
    data_quality = _extract_data_quality(prediction)
    artifact = data.get("artifact_quarantine") or {}
    health = data.get("report_health_gate") or {}
    contract = data.get("data_contract") or {}
    training = data.get("training_status") or {}
    cards = "".join(
        [
            _metric_card("Live Betting", governance.get("live_betting_allowed", False), "must remain disabled", "ok" if governance.get("live_betting_allowed") is False else "bad"),
            _metric_card("Model Mode", governance.get("mode"), "active serving policy", "warn"),
            _metric_card("Artifact", artifact.get("status"), "stale artifact quarantine", artifact.get("status")),
            _metric_card("Pipeline", health.get("pipeline_health_status", health.get("status")), "health gate", health.get("pipeline_health_status", health.get("status"))),
            _metric_card("Data Contract", contract.get("status"), f"errors={contract.get('error_count', 0)} warnings={contract.get('warning_count', 0)}", contract.get("status")),
            _metric_card("Training", training.get("trained"), f"samples={training.get('sample_count')} min={training.get('minimum_clean_train_samples')}", "warn" if not training.get("trained") else "ok"),
        ]
    )
    detail = "".join(
        [
            _details("Governance block reasons", _list_items(governance.get("block_reasons") or []), True),
            _details("Missing critical sources", _list_items(data_quality.get("missing_critical_sources") or [])),
            _details("Missing important sources", _list_items(data_quality.get("missing_important_sources") or [])),
            _details("Model quality blocks", _list_items(health.get("model_quality_blocks") or [])),
        ]
    )
    return f"<section class='panel'><div class='section-title'><h2>Governance & Safety</h2><span>Why the system is locked or allowed</span></div><div class='metric-grid'>{cards}</div>{detail}</section>"


def _prediction_cards(report: Optional[Dict[str, Any]]) -> str:
    games = _predictions(report)
    if not games:
        return "<p class='muted'>No prediction cards available.</p>"
    cards: list[str] = []
    for item in games[:24]:
        home = item.get("home_team") or item.get("home") or "Home"
        away = item.get("away_team") or item.get("away") or "Away"
        recommendation = item.get("recommendation") or item.get("moneyline_recommendation") or "TRACKING ONLY"
        probability = item.get("predicted_home_win_pct") or item.get("displayed_home_win_pct") or item.get("home_win_probability")
        status = item.get("recommendation_status") or item.get("signal_status") or "tracking"
        quality = item.get("data_quality_status") if isinstance(item.get("data_quality_status"), dict) else {}
        cards.append(
            "<article class='game-card'>"
            f"<div class='card-top'>{_badge(status, status)}<span>{_escape(item.get('game_date') or item.get('start_time'))}</span></div>"
            f"<h3>{_escape(away)} <span>@</span> {_escape(home)}</h3>"
            f"<div class='prob'>{_value(probability)}<small>home win probability</small></div>"
            f"<p>{_escape(recommendation)}</p>"
            f"<small>Data grade: {_escape(quality.get('data_quality_grade'))}</small>"
            "</article>"
        )
    return "<div class='game-grid'>" + "".join(cards) + "</div>"


def build_html() -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = {key: _read_json(path) for key, path in SOURCES.items()}
    title, title_status, reasons = _topline_status(data)
    prediction = data.get("prediction")
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    summary_cards = "".join(
        [
            _metric_card("Predictions", len(_predictions(prediction)), "today board size", "neutral"),
            _metric_card("Promotion", (data.get("baseline_comparison") or {}).get("promotion_allowed"), "baseline quality gate", "warn"),
            _metric_card("Model Quality", (data.get("report_health_gate") or {}).get("model_quality_status"), "quality blocks are not pipeline crashes", (data.get("report_health_gate") or {}).get("model_quality_status")),
            _metric_card("Feature Gaps", (data.get("feature_priority") or {}).get("p1_high_impact_data_gap_count"), "P1 high-impact data gaps", "warn"),
        ]
    )

    html_parts = [
        "<!doctype html>",
        "<html lang='zh-Hant'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>MLB Intelligence Cloud · Research Dashboard</title>",
        "<style>",
        ":root{color-scheme:dark;--bg:#07100d;--panel:#0f1f1a;--panel2:#132821;--text:#f3fff9;--muted:#a7c7b8;--line:#2c4b40;--ok:#41e69f;--warn:#f6c453;--bad:#ff7b93;--neutral:#91c8ff;--shadow:0 18px 50px rgba(0,0,0,.38);}",
        "*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top left,rgba(65,230,159,.18),transparent 34%),linear-gradient(180deg,#07100d,#050806);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,Arial,sans-serif;font-variant-numeric:tabular-nums;line-height:1.5}",
        ".shell{max-width:1280px;margin:0 auto;padding:22px 16px 44px}.hero{border:1px solid var(--line);background:linear-gradient(135deg,rgba(19,40,33,.96),rgba(8,18,15,.96));box-shadow:var(--shadow);border-radius:28px;padding:28px;margin-bottom:16px}.eyebrow{color:var(--ok);font-size:.75rem;letter-spacing:.18em;text-transform:uppercase;font-weight:850}.hero h1{font-size:clamp(2rem,5vw,4rem);letter-spacing:-.065em;line-height:1;margin:.4rem 0}.hero p{max-width:820px;color:var(--muted);margin:.5rem 0 0}.status-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}",
        ".badge{display:inline-flex;align-items:center;gap:6px;padding:7px 10px;border-radius:999px;font-size:.72rem;font-weight:850;text-transform:uppercase;border:1px solid var(--line);background:#0b1713;color:var(--muted)}.badge.ok{color:#06120d;background:var(--ok);border-color:var(--ok)}.badge.warn{color:#1d1300;background:var(--warn);border-color:var(--warn)}.badge.bad{color:#2d0007;background:var(--bad);border-color:var(--bad)}.badge.neutral{color:#061427;background:var(--neutral);border-color:var(--neutral)}",
        ".panel{border:1px solid var(--line);background:rgba(15,31,26,.88);border-radius:22px;padding:18px;margin:14px 0;box-shadow:0 12px 32px rgba(0,0,0,.22)}.section-title{display:flex;justify-content:space-between;gap:12px;align-items:end;margin-bottom:12px}.section-title h2{margin:0;font-size:1.15rem}.section-title span{color:var(--muted);font-size:.78rem}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.metric-card{background:#0b1713;border:1px solid var(--line);border-radius:18px;padding:14px;min-height:112px}.metric-card.ok{border-color:rgba(65,230,159,.5)}.metric-card.warn{border-color:rgba(246,196,83,.55)}.metric-card.bad{border-color:rgba(255,123,147,.6)}.metric-label{color:var(--muted);font-size:.67rem;text-transform:uppercase;letter-spacing:.12em;font-weight:850}.metric-value{font-size:1.45rem;font-weight:920;margin-top:6px}.metric-caption{color:var(--muted);font-size:.72rem;margin-top:4px}",
        ".details{border:1px solid var(--line);border-radius:16px;margin-top:10px;background:rgba(5,10,8,.45);padding:0}.details summary{cursor:pointer;padding:12px 14px;font-weight:850;color:var(--text)}.details>*:not(summary){padding:0 14px 14px}.clean-list{margin:.25rem 0 0;padding-left:1.2rem;color:var(--muted)}.muted{color:var(--muted)}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:14px;margin-top:10px}table{width:100%;border-collapse:collapse;min-width:720px}th,td{padding:10px;border-bottom:1px solid rgba(255,255,255,.08);text-align:left;font-size:.82rem}th{background:#132821;color:#dffff2;text-transform:uppercase;font-size:.68rem;letter-spacing:.1em}td{color:#e8fff5}",
        ".action-grid,.game-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:12px 0}.action-card,.game-card{background:#0b1713;border:1px solid var(--line);border-radius:16px;padding:14px}.action-card p,.game-card p{color:var(--muted);margin:.45rem 0}.action-card small,.game-card small{color:#89ac9e}.card-top{display:flex;justify-content:space-between;gap:10px;align-items:center;color:var(--muted);font-size:.72rem}.game-card h3{margin:.7rem 0 .4rem;font-size:1rem}.game-card h3 span{color:var(--muted)}.prob{font-size:1.55rem;font-weight:920}.prob small{display:block;font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}footer{color:var(--muted);text-align:center;margin-top:24px;font-size:.75rem}@media(max-width:980px){.metric-grid,.action-grid,.game-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:640px){.shell{padding:12px}.hero{padding:20px;border-radius:20px}.metric-grid,.action-grid,.game-grid{grid-template-columns:1fr}.section-title{display:block}table{min-width:620px}}",
        "</style>",
        "</head>",
        "<body><main class='shell'>",
        "<section class='hero'>",
        "<div class='eyebrow'>MLB Intelligence Cloud · Paper-only research dashboard</div>",
        f"<h1>{_escape(title)}</h1>",
        "<p>先看模型是否可信，再看單場預測。這個介面把 pipeline 健康、模型品質、baseline、artifact quarantine、feature roadmap 分開顯示，避免把『模型不能升級』誤看成『程式壞掉』。</p>",
        "<div class='status-row'>" + _badge(title, title_status) + _badge("Live betting locked", "ok") + _badge("No automated wagering", "ok") + _badge(f"Generated {generated_at}", "neutral") + "</div>",
        _list_items(reasons, "No active block reasons."),
        "</section>",
        f"<section class='panel'><div class='section-title'><h2>Executive Snapshot</h2><span>Most important information first</span></div><div class='metric-grid'>{summary_cards}</div></section>",
        _governance_section(data),
        _model_quality_section(data),
        _feature_priority_section(data.get("feature_priority")),
        f"<section class='panel'><div class='section-title'><h2>Game Board Preview</h2><span>Today's cards, still paper-only</span></div>{_prediction_cards(prediction)}</section>",
        "<section class='panel'><div class='section-title'><h2>Risk Disclosure</h2><span>Required safety copy</span></div><p class='muted'>This project is for research and educational use only. It does not provide financial advice, gambling advice, guaranteed profit, or betting instructions. Sports betting is risky and markets can be highly efficient. Paper trading is the default and live betting is disabled by governance.</p></section>",
        f"<footer>Generated at {_escape(generated_at)} · Model promotion, live betting and automated wagering remain disabled.</footer>",
        "</main></body></html>",
    ]
    html_content = "\n".join(html_parts)
    OUTPUT_HTML.write_text(html_content, encoding="utf-8")
    return html_content


def main() -> None:
    build_html()
    print(json.dumps({"status": "ok", "output_path": str(OUTPUT_HTML)}, indent=2))


if __name__ == "__main__":
    main()
