from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

REPORT_DIR = Path("report")
OUTPUT_HTML = REPORT_DIR / "index.html"
SITE_DATA_PATH = Path("site/data/public_dashboard.json")


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(v) for v in value]
    return str(value)


def _load_dashboard() -> dict[str, Any]:
    try:
        from scripts.build_public_site_data import build_public_dashboard

        payload = build_public_dashboard()
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    try:
        payload = json.loads(SITE_DATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _e(value: Any, fallback: str = "--") -> str:
    if value is None or value == "":
        value = fallback
    return html.escape(str(value))


def _num(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _pct(value: Any, ratio: bool = False) -> str:
    parsed = _num(value)
    if parsed is None:
        return "--"
    if ratio:
        parsed *= 100.0
    return f"{parsed:.1f}%"


def _signed_pct(value: Any) -> str:
    parsed = _num(value)
    if parsed is None:
        return "--"
    sign = "+" if parsed > 0 else ""
    return f"{sign}{parsed:.1f}%"


def _int(value: Any) -> str:
    parsed = _num(value)
    return "--" if parsed is None else str(int(round(parsed)))


def _status_class(value: Any) -> str:
    status = str(value or "").lower()
    if any(word in status for word in ["failed", "error", "fatal", "danger"]):
        return "danger"
    if any(word in status for word in ["blocked", "warning", "quarantined", "insufficient", "tracking"]):
        return "warning"
    if any(word in status for word in ["ok", "completed", "ready"]):
        return "ok"
    return "neutral"


def _is_paper_signal(game: dict[str, Any]) -> bool:
    raw = str(game.get("recommendation_raw") or "").upper()
    return bool(raw and raw not in {"NO BET", "TRACKING ONLY"})


def _side_context(game: dict[str, Any]) -> dict[str, Any]:
    home_prob = _num(game.get("home_win_probability_pct"))
    away_prob = None if home_prob is None else 100.0 - home_prob
    home_edge = _num(game.get("model_edge_home_pct"))
    away_edge = None if home_edge is None else -home_edge
    if home_prob is None:
        projected = None
    elif home_prob >= 50:
        projected = {"team": game.get("home_team"), "prob": home_prob}
    else:
        projected = {"team": game.get("away_team"), "prob": away_prob}
    if home_edge is None:
        lean = None
    elif home_edge >= 0:
        lean = {"team": game.get("home_team"), "edge": home_edge}
    else:
        lean = {"team": game.get("away_team"), "edge": away_edge}
    return {"home_prob": home_prob, "away_prob": away_prob, "home_edge": home_edge, "away_edge": away_edge, "projected": projected, "lean": lean}


def _metric(label: str, value: Any, caption: str) -> str:
    return f"""
    <article class="metric-card">
      <span>{_e(label)}</span>
      <strong>{_e(value)}</strong>
      <small>{_e(caption)}</small>
    </article>
    """


def _game_card(game: dict[str, Any]) -> str:
    side = _side_context(game)
    projected = "--" if not side["projected"] else f"{_e(side['projected'].get('team'))} {_pct(side['projected'].get('prob'))}"
    lean = "--" if not side["lean"] else f"{_e(side['lean'].get('team'))} {_signed_pct(side['lean'].get('edge'))}"
    paper_tag = _e(game.get("recommendation_raw") if _is_paper_signal(game) else "No Bet")
    flags = "".join(f"<span class='tag'>{_e(flag)}</span>" for flag in (game.get("risk_flags") or [])[:5])
    status_class = _status_class(f"{game.get('recommendation_status')} {game.get('risk_profile')}")
    pitchers = f"{_e(game.get('away_probable_pitcher_name'), 'Away pitcher TBD')} vs {_e(game.get('home_probable_pitcher_name'), 'Home pitcher TBD')}"
    notes = game.get("public_notes") if isinstance(game.get("public_notes"), list) else []
    note = notes[0] if notes else "Tracking only. No betting recommendation."
    return f"""
    <article class="game-card">
      <div class="game-mainline">
        <div>
          <div class="game-top"><span class="status-pill {status_class}">{_e(game.get('recommendation_label'), 'TRACKING ONLY')}</span><span>{_e(game.get('start_time'))}</span></div>
          <h3>{_e(game.get('away_team'))} <span>@</span> {_e(game.get('home_team'))}</h3>
          <p class="match-meta">{pitchers}<br>{_e(game.get('moneyline_gate_status'), 'tracking gate')}</p>
        </div>
        <div>
          <div class="prob-row">
            <div class="prob-box"><strong>{projected}</strong><span>Projected side</span></div>
            <div class="prob-box"><strong>{lean}</strong><span>Value lean vs market</span></div>
          </div>
          <p class="game-note">{_e(note)}<br><small>Home {_pct(side['home_prob'])} · Away {_pct(side['away_prob'])}</small><br><small>Market home {_pct(game.get('market_home_probability_pct'))} · Home edge {_signed_pct(side['home_edge'])}</small></p>
        </div>
      </div>
      <div class="tag-row"><span class="tag strong">{paper_tag}</span><span class="tag">Grade {_e(game.get('data_quality_grade'))}</span><span class="tag">{_e(game.get('lineup_status'), 'lineup unknown')}</span><span class="tag">{_e(game.get('pitcher_status'), 'pitcher unknown')}</span>{flags}</div>
    </article>
    """


def _roadmap_card(action: dict[str, Any]) -> str:
    features = "".join(f"<li>{_e(feature)}</li>" for feature in (action.get("top_features") or [])[:5])
    return f"""
    <article class="roadmap-card">
      <strong>{_e(action.get('feature_group'), 'Feature group')}</strong>
      <p>{_e(action.get('rationale'), 'Review and backfill this feature group.')}</p>
      <ul>{features}</ul>
    </article>
    """


def build_html(data: dict[str, Any]) -> str:
    system = data.get("system_status") if isinstance(data.get("system_status"), dict) else {}
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    performance = data.get("performance") if isinstance(data.get("performance"), dict) else {}
    official = performance.get("official_accuracy") if isinstance(performance.get("official_accuracy"), dict) else {}
    rolling = performance.get("rolling_windows") if isinstance(performance.get("rolling_windows"), dict) else {}
    slices = performance.get("slices") if isinstance(performance.get("slices"), dict) else {}
    pending = performance.get("pending_predictions") if isinstance(performance.get("pending_predictions"), dict) else {}
    clv = performance.get("clv_metrics") if isinstance(performance.get("clv_metrics"), dict) else {}
    governance = data.get("governance_summary") if isinstance(data.get("governance_summary"), dict) else {}
    artifact = governance.get("artifact_quarantine") if isinstance(governance.get("artifact_quarantine"), dict) else {}
    games = data.get("games") if isinstance(data.get("games"), list) else []
    actions = (data.get("feature_roadmap") or {}).get("actions") if isinstance(data.get("feature_roadmap"), dict) else []
    actions = actions if isinstance(actions, list) else []
    paper_signals = sum(1 for game in games if isinstance(game, dict) and _is_paper_signal(game))
    no_bets = max(0, len(games) - paper_signals)

    stylesheet = """
:root{color-scheme:dark;--bg:#071019;--panel:rgba(14,25,38,.92);--line:rgba(135,166,210,.18);--text:#f8fbff;--muted:#a7b6c9;--muted2:#6f8196;--blue:#74c0fc;--green:#53e08f;--yellow:#ffd166;--red:#ff6b6b;--shadow:0 24px 80px rgba(0,0,0,.34)}*{box-sizing:border-box}body{margin:0;color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-variant-numeric:tabular-nums;background:radial-gradient(circle at 18% -8%,rgba(83,224,143,.18),transparent 30%),radial-gradient(circle at 88% 6%,rgba(116,192,252,.20),transparent 34%),linear-gradient(180deg,#071019,#0b121d 55%,#05080d)}.shell{max-width:1180px;margin:0 auto;padding:24px 18px 58px}.hero{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(290px,.6fr);gap:16px}.card,.panel,.notice,.metric-card,.game-card,.roadmap-card{border:1px solid var(--line);background:var(--panel);box-shadow:var(--shadow)}.hero-main{min-height:360px;border-radius:34px;padding:clamp(26px,6vw,56px);display:flex;flex-direction:column;justify-content:center;background:linear-gradient(140deg,rgba(17,34,52,.96),rgba(9,18,28,.90))}.hero-side{border-radius:34px;padding:22px;background:linear-gradient(180deg,rgba(18,32,48,.98),rgba(8,15,24,.94));display:flex;flex-direction:column;justify-content:space-between}.eyebrow{margin:0 0 10px;color:var(--green);text-transform:uppercase;font-size:.74rem;font-weight:900;letter-spacing:.16em}.hero h1{margin:0;font-size:clamp(2.4rem,7vw,5.45rem);line-height:.94;letter-spacing:-.075em}.hero-subtitle{color:var(--muted);font-size:1.05rem;line-height:1.7;margin:20px 0 0}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:30px}.button{display:inline-flex;min-height:44px;align-items:center;justify-content:center;padding:0 16px;border-radius:999px;text-decoration:none;font-weight:900}.button.primary{background:var(--text);color:#08111d}.button.secondary{background:rgba(83,224,143,.14);color:var(--green);border:1px solid rgba(83,224,143,.34)}.button.ghost{color:var(--text);border:1px solid var(--line);background:rgba(255,255,255,.055)}.status-pill{display:inline-flex;width:max-content;padding:9px 12px;border-radius:999px;font-weight:950;font-size:.76rem;text-transform:uppercase;letter-spacing:.08em}.status-pill.ok{background:rgba(83,224,143,.16);color:var(--green);border:1px solid rgba(83,224,143,.35)}.status-pill.warning{background:rgba(255,209,102,.16);color:var(--yellow);border:1px solid rgba(255,209,102,.35)}.status-pill.danger{background:rgba(255,107,107,.16);color:var(--red);border:1px solid rgba(255,107,107,.35)}.status-pill.neutral{background:rgba(116,192,252,.16);color:var(--blue);border:1px solid rgba(116,192,252,.35)}.quick-status{margin:28px 0 0;display:grid;gap:10px}.quick-status div{display:flex;justify-content:space-between;gap:16px;padding:12px 0;border-bottom:1px solid rgba(255,255,255,.08)}.quick-status dt{color:var(--muted)}.quick-status dd{margin:0;font-weight:900;text-align:right}.notice{margin:16px 0;border-radius:18px;padding:14px 16px;display:flex;gap:10px;color:var(--muted)}.notice strong{color:var(--yellow)}.score-strip{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:16px 0 26px}.metric-card{min-height:116px;border-radius:20px;padding:16px;background:linear-gradient(180deg,rgba(17,31,47,.94),rgba(9,16,26,.90))}.metric-card span,.record-item span,.performance-item span{display:block;color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;font-weight:900}.metric-card strong,.record-item strong,.performance-item strong{display:block;margin-top:10px;font-size:1.8rem;letter-spacing:-.035em}.metric-card small,.record-item small,.performance-item small{display:block;color:var(--muted2);margin-top:5px}.section{margin-top:32px}.section-heading{display:flex;justify-content:space-between;align-items:end;gap:14px;margin-bottom:14px}.section-heading h2{margin:0;font-size:clamp(1.45rem,3vw,2.2rem);letter-spacing:-.045em}.section-note{color:var(--muted);font-size:.9rem}.game-list{display:grid;gap:12px}.game-card{border-radius:22px;padding:16px;background:linear-gradient(180deg,rgba(17,31,47,.95),rgba(8,15,24,.90))}.game-mainline{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(260px,.85fr);gap:14px}.game-top{display:flex;justify-content:space-between;align-items:center;gap:10px;color:var(--muted);font-size:.82rem}.game-card h3{margin:14px 0 8px;font-size:1.35rem;letter-spacing:-.03em}.game-card h3 span{color:var(--muted2);font-weight:500}.match-meta,.game-note{color:var(--muted);line-height:1.55}.prob-row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.prob-box{background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:12px}.prob-box strong{display:block;font-size:1.45rem;letter-spacing:-.035em}.prob-box span{color:var(--muted);font-size:.75rem}.tag-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.tag{border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.055);color:var(--muted);padding:6px 8px;border-radius:999px;font-size:.75rem}.tag.strong{color:var(--green);border-color:rgba(83,224,143,.25);background:rgba(83,224,143,.1)}.two-column{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{border-radius:24px;padding:20px}.record-board,.performance-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.record-item,.performance-item{border:1px solid var(--line);background:rgba(255,255,255,.035);border-radius:18px;padding:16px}.record-wide{grid-column:1/-1}.roadmap-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.roadmap-card{border-radius:22px;padding:18px;box-shadow:none}.roadmap-card p{color:var(--muted);line-height:1.55}.roadmap-card ul{margin:12px 0 0;padding-left:18px;color:var(--muted)}@media(max-width:1040px){.hero,.two-column,.game-mainline{grid-template-columns:1fr}.score-strip{grid-template-columns:repeat(3,minmax(0,1fr))}.roadmap-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:680px){.shell{padding:16px 12px 42px}.hero-main,.hero-side{border-radius:24px}.score-strip,.record-board,.performance-grid,.roadmap-grid{grid-template-columns:1fr}.section-heading{align-items:flex-start;flex-direction:column}.notice{flex-direction:column}.prob-row{grid-template-columns:1fr}}
    """

    game_cards = "".join(_game_card(game) for game in games if isinstance(game, dict)) or "<div class='notice'>目前沒有可顯示的賽事。</div>"
    roadmap_cards = "".join(_roadmap_card(action) for action in actions if isinstance(action, dict)) or "<div class='notice'>Feature roadmap 尚未產出。</div>"

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLB Paper Prediction Board</title>
<style>{stylesheet}</style>
</head>
<body>
<main class="shell">
  <section class="hero">
    <div class="hero-main card">
      <p class="eyebrow">AI Research · Daily MLB Board · Governance Locked</p>
      <h1>MLB Paper Prediction Board</h1>
      <p class="hero-subtitle">每日 MLB 賽事重點整理、paper-only 訊號、已結算戰績與模型治理狀態。</p>
      <div class="actions"><a href="#games" class="button primary">今日賽事</a><a href="#record" class="button secondary">預測戰績</a><a href="#governance" class="button ghost">模型治理</a></div>
    </div>
    <aside class="hero-side card">
      <span class="status-pill warning">Research mode · tracking only</span>
      <dl class="quick-status"><div><dt>Pipeline</dt><dd>{_e(system.get('pipeline_health_status'))}</dd></div><div><dt>Model</dt><dd>{_e(system.get('model_quality_status'))}</dd></div><div><dt>Data Contract</dt><dd>{_e(system.get('data_contract_status'))}</dd></div><div><dt>Updated</dt><dd>{_e(data.get('source_generated_at') or data.get('generated_at'))}</dd></div></dl>
    </aside>
  </section>
  <section class="notice"><strong>安全聲明</strong><span>{_e(data.get('public_disclaimer'), 'Research dashboard only. No betting advice, no automated wagering, and no live-betting enablement.')}</span></section>
  <section class="score-strip">
    {_metric('今日場次', metrics.get('game_count') or metrics.get('scheduled_game_count') or len(games), 'scheduled games')}
    {_metric('Paper signals', paper_signals, 'tracking-only leans')}
    {_metric('No Bet', no_bets, 'blocked / watch only')}
    {_metric('已結算樣本', official.get('sample_count') or metrics.get('clean_settled_sample_count'), 'trusted outcomes')}
    {_metric('30日命中率', _pct((rolling.get('30d') or {}).get('accuracy'), ratio=True), 'settled only')}
    {_metric('CLV+', _pct(clv.get('positive_clv_rate') or (metrics.get('positive_clv_rate_pct') or 0) / 100, ratio=True), 'price movement')}
  </section>
  <section id="games" class="section"><div class="section-heading"><div><p class="eyebrow">Today</p><h2>今日 MLB 預測</h2></div><span class="section-note">全部為 paper / tracking-only，不提供下注指令。</span></div><div class="game-list">{game_cards}</div></section>
  <section id="record" class="section two-column"><div class="panel"><div class="section-heading"><div><p class="eyebrow">Record</p><h2>預測戰績</h2></div></div><div class="record-board">
    <article class="record-item record-wide"><span>✓ / ✗</span><strong>{_int(official.get('correct'))} / {_int((_num(official.get('sample_count')) or 0) - (_num(official.get('correct')) or 0))}</strong><small>official settled record</small></article>
    <article class="record-item"><span>Official accuracy</span><strong>{_pct(official.get('accuracy'), ratio=True)}</strong><small>{_int(official.get('sample_count'))} settled games</small></article>
    <article class="record-item"><span>7D accuracy</span><strong>{_pct((rolling.get('7d') or {}).get('accuracy'), ratio=True)}</strong><small>{_int((rolling.get('7d') or {}).get('sample_count'))} samples</small></article>
    <article class="record-item"><span>30D accuracy</span><strong>{_pct((rolling.get('30d') or {}).get('accuracy'), ratio=True)}</strong><small>{_int((rolling.get('30d') or {}).get('sample_count'))} samples</small></article>
    <article class="record-item"><span>Pending</span><strong>{_int(pending.get('count'))}</strong><small>{_e(pending.get('reason'), 'unsettled')}</small></article>
    <article class="record-item"><span>Paper signals</span><strong>{_pct((slices.get('paper_signals') or {}).get('accuracy'), ratio=True)}</strong><small>{_int((slices.get('paper_signals') or {}).get('sample_count'))} samples</small></article>
  </div></div><div class="panel"><div class="section-heading"><div><p class="eyebrow">Performance</p><h2>模型品質</h2></div></div><div class="performance-grid">
    <article class="performance-item"><span>Brier</span><strong>{_e(official.get('brier') or metrics.get('model_brier'))}</strong><small>lower is better</small></article>
    <article class="performance-item"><span>Logloss</span><strong>{_e(official.get('logloss'))}</strong><small>probability quality</small></article>
    <article class="performance-item"><span>Model AUC</span><strong>{_e(metrics.get('model_auc'))}</strong><small>current eval</small></article>
    <article class="performance-item"><span>CLV avg</span><strong>{_e(clv.get('avg_clv') or metrics.get('avg_clv'))}</strong><small>price movement only</small></article>
  </div></div></section>
  <section id="governance" class="section two-column"><div class="panel"><p class="eyebrow">Governance</p><h2>模型治理狀態</h2><p class="match-meta">Pipeline: {_e(system.get('pipeline_health_status'))}<br>Model: {_e(system.get('model_quality_status'))}<br>Live betting: {_e(system.get('live_betting_allowed'))}<br>Automated wagering: {_e(system.get('automated_wagering_allowed'))}</p></div><div class="panel"><p class="eyebrow">Artifact</p><h2>模型 Artifact 鎖定</h2><p class="match-meta">Status: {_e(artifact.get('status'))}<br>Artifact samples: {_e(artifact.get('artifact_training_sample_count'))}<br>Current samples: {_e(artifact.get('current_training_sample_count'))}<br>Production minimum: {_e(artifact.get('minimum_production_training_samples'))}</p></div></section>
  <section id="roadmap" class="section"><div class="section-heading"><div><p class="eyebrow">Roadmap</p><h2>資料源修補優先順序</h2></div><span class="section-note">先修資料，不急著升級模型。</span></div><div class="roadmap-grid">{roadmap_cards}</div></section>
</main>
</body>
</html>"""


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_dashboard()
    OUTPUT_HTML.write_text(build_html(_safe(data)), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_path": str(OUTPUT_HTML), "site_version": data.get("site_version")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
