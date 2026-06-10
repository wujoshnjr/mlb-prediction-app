# main.py
"""MLB Intelligence Cloud FastAPI dashboard.

Product UI Upgrade v1:
- Mission Control
- Research Guardrails
- Signal Center
- Product-style Game Board
- Game Detail payloads

Safety: paper/shadow research only. No live betting, no automated wagering,
no user funds, and no production model replacement.
"""

from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sklearn.metrics import brier_score_loss

try:
    from prediction import generate_predictions
except Exception as exc:  # pragma: no cover
    print(f"Warning: prediction module failed to import: {exc}")
    generate_predictions = None


app = FastAPI(title="MLB Intelligence Cloud")
ADMIN_TOKEN = os.getenv("ADMIN_API_TOKEN", "")

REPORT_PATH = Path("report/prediction.json")
SNAPSHOT_PATH = Path("data/prediction_snapshots.csv")
MARKET_ODDS_PATH = Path("data/market_odds_history.csv")
DAILY_CONTEXT_PATH = Path("data/daily_game_context.csv")
TRAINING_STATUS_PATH = Path("data/training_status.json")
SAMPLE_STATE_PATH = Path("data/sample_state.json")
FINALIZED_GAMES_PATH = Path("data/finalized_games.csv")
FINALIZED_SNAPSHOT_OUTCOMES_PATH = Path("data/finalized_snapshot_outcomes.csv")
LINEUP_QUALITY_CONTEXT_PATH = Path("data/lineup_quality_context.csv")
CLEAN_PIPELINE_VERSION = "baseline_v2_clean"

HTML = r"""
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>MLB Intelligence Cloud</title>
<style>
:root{color-scheme:dark;--bg:#020807;--panel:rgba(7,24,20,.86);--border:rgba(52,211,153,.18);--text:#ecfff8;--muted:#8eb7a7;--muted2:#5d7f72;--green:#34f5a4;--cyan:#22d3ee;--amber:#fbbf24;--red:#fb7185;--greenbg:rgba(52,245,164,.12);--amberbg:rgba(251,191,36,.12);--redbg:rgba(251,113,133,.12);--grad:linear-gradient(92deg,#00ff88,#22d3ee 52%,#a7f3d0)}
*{box-sizing:border-box}body{margin:0;min-height:100%;background:radial-gradient(circle at 12% -8%,rgba(52,245,164,.22),transparent 34%),radial-gradient(circle at 88% 0%,rgba(34,211,238,.13),transparent 31%),linear-gradient(180deg,#020807,#04130f 46%,#020807);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Arial,sans-serif;font-variant-numeric:tabular-nums;overflow-x:hidden}body:before{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;background-image:linear-gradient(rgba(52,211,153,.055) 1px,transparent 1px),linear-gradient(90deg,rgba(52,211,153,.045) 1px,transparent 1px);background-size:44px 44px}.container{max-width:1380px;margin:0 auto;padding:22px 18px 38px}.topbar{display:flex;justify-content:space-between;align-items:center;gap:14px;margin-bottom:18px;padding:11px 12px;border:1px solid rgba(134,239,172,.10);border-radius:20px;background:rgba(2,8,7,.52);backdrop-filter:blur(16px)}.brand{display:flex;align-items:center;gap:12px}.brand-mark{display:grid;place-items:center;width:46px;height:46px;border-radius:15px;background:linear-gradient(140deg,rgba(52,245,164,.96),rgba(34,211,238,.72));color:#03110e;font-weight:950}.brand-name{font-size:1.03rem;font-weight:780}.brand-sub{color:var(--muted);font-size:.68rem;letter-spacing:.18em;margin-top:3px;text-transform:uppercase}.live-pill{display:flex;align-items:center;gap:8px;border:1px solid rgba(52,245,164,.34);background:rgba(52,245,164,.10);color:var(--green);border-radius:999px;padding:9px 12px;font-size:.72rem;font-weight:780;text-transform:uppercase}.live-dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 14px var(--green)}.hero{overflow:hidden;padding:26px 24px;margin-bottom:16px;border-radius:26px;background:linear-gradient(120deg,rgba(8,35,28,.96),rgba(3,17,14,.94));border:1px solid var(--border);box-shadow:0 24px 70px rgba(0,0,0,.42)}.hero-content{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(270px,.75fr);gap:22px;align-items:end}.hero-kicker{color:#b9ffe1;font-size:.69rem;font-weight:800;letter-spacing:.19em;text-transform:uppercase;margin-bottom:13px}.hero h1{margin:0;font-size:clamp(1.9rem,4.2vw,3.25rem);font-weight:860;letter-spacing:-.06em;line-height:1.02}.hero-gradient{background:var(--grad);background-clip:text;-webkit-background-clip:text;color:transparent}.hero-copy{color:var(--muted);max-width:760px;margin:14px 0 0;line-height:1.58;font-size:.92rem}.updated{display:inline-flex;margin-top:18px;padding:8px 12px;border-radius:999px;border:1px solid rgba(52,211,153,.17);background:rgba(255,255,255,.035);color:#b4d7ca;font-size:.72rem}.terminal-card{padding:16px;border-radius:20px;border:1px solid rgba(52,245,164,.18);background:linear-gradient(150deg,rgba(2,8,7,.52),rgba(6,25,21,.62))}.terminal-title{display:flex;justify-content:space-between;gap:12px;font-size:.72rem;color:var(--muted);letter-spacing:.14em;text-transform:uppercase;margin-bottom:10px}.terminal-status{color:var(--green);font-weight:900}.terminal-line{display:flex;justify-content:space-between;gap:14px;border-top:1px solid rgba(52,211,153,.09);padding:10px 0;color:#d9fff1;font-size:.78rem}.terminal-line span:first-child{color:var(--muted)}.section-heading{display:flex;justify-content:space-between;align-items:end;gap:12px;margin:20px 0 10px}.section-heading h2{margin:0;font-size:1.02rem}.section-heading span{color:var(--muted);font-size:.72rem}.messages{display:grid;gap:8px;margin:12px 0}.message{padding:11px 13px;border-radius:14px;border:1px solid rgba(52,211,153,.12);background:rgba(7,24,20,.70);color:#c8f7e4;font-size:.72rem}.message.warn{color:var(--amber);background:var(--amberbg)}.message.bad{color:var(--red);background:var(--redbg)}.stats,.health-grid,.guardrail-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}.health-grid,.guardrail-grid{grid-template-columns:repeat(4,minmax(0,1fr))}.stat,.health-card,.guardrail-card{min-height:104px;padding:14px;border-radius:18px;border:1px solid rgba(52,211,153,.12);background:rgba(7,24,20,.72);box-shadow:0 16px 42px rgba(0,0,0,.22)}.stat.featured{background:linear-gradient(145deg,rgba(52,245,164,.13),rgba(34,211,238,.06),rgba(7,24,20,.72))}.stat-label,.health-label,.guardrail-label{color:var(--muted);font-size:.62rem;font-weight:820;letter-spacing:.13em;text-transform:uppercase;margin-bottom:8px}.stat-value,.health-value,.guardrail-value{font-size:1.48rem;font-weight:900;letter-spacing:-.04em}.positive{color:var(--green)}.negative{color:var(--red)}.waiting{color:var(--amber)}.neutral{color:var(--text)}.stat-caption,.health-caption,.guardrail-caption{color:var(--muted2);font-size:.64rem;line-height:1.35;margin-top:7px}.signal-center{padding:14px;border:1px solid rgba(52,211,153,.14);background:rgba(7,24,20,.72);border-radius:18px;box-shadow:0 16px 42px rgba(0,0,0,.22)}.case-tabs{display:flex;flex-wrap:wrap;gap:8px}.case-tab{cursor:pointer;border:1px solid rgba(52,211,153,.15);color:#c8f7e4;background:rgba(2,8,7,.45);border-radius:999px;padding:8px 10px;font-size:.66rem;font-weight:840;letter-spacing:.06em;text-transform:uppercase}.case-tab.active{background:linear-gradient(92deg,rgba(52,245,164,.92),rgba(34,211,238,.70));color:#02110d}.games{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.game-card{overflow:hidden;border-radius:20px;border:1px solid rgba(52,211,153,.13);background:rgba(7,24,20,.82);box-shadow:0 18px 48px rgba(0,0,0,.28)}.game-card.product{cursor:pointer;transition:transform .18s ease,border-color .18s ease}.game-card.product:hover{transform:translateY(-2px);border-color:rgba(52,245,164,.42)}.signal{min-height:46px;display:flex;align-items:center;justify-content:space-between;gap:9px;padding:11px 14px;color:#c7ffe7;font-size:.71rem;font-weight:840;text-transform:uppercase;letter-spacing:.09em;border-bottom:1px solid rgba(52,211,153,.10);background:linear-gradient(92deg,rgba(52,245,164,.12),rgba(34,211,238,.06))}.signal.paper{background:linear-gradient(92deg,rgba(52,245,164,.95),rgba(34,211,238,.76));color:#02110d}.signal.blocked{background:linear-gradient(92deg,rgba(251,113,133,.30),rgba(3,17,14,.36));color:#ffd1d8}.signal.no-signal{background:rgba(142,183,167,.08);color:#b1d4c7}.signal-edge{font-size:.69rem;white-space:nowrap}.game-body{padding:15px}.matchup-row{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:14px}.matchup{font-size:1.04rem;font-weight:820}.game-time{color:var(--muted);font-size:.71rem;margin-top:6px}.probability{text-align:right;font-weight:900;font-size:1.43rem;line-height:1}.probability small{display:block;font-size:.58rem;letter-spacing:.10em;color:var(--muted);margin-top:6px;text-transform:uppercase}.tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}.tag{display:inline-flex;align-items:center;padding:6px 8px;border-radius:999px;font-size:.61rem;line-height:1;font-weight:820;letter-spacing:.07em;text-transform:uppercase}.tag.green{background:var(--greenbg);color:var(--green);border:1px solid rgba(52,245,164,.14)}.tag.amber{background:var(--amberbg);color:var(--amber);border:1px solid rgba(251,191,36,.13)}.tag.red{background:var(--redbg);color:var(--red);border:1px solid rgba(251,113,133,.15)}.tag.muted{background:rgba(142,183,167,.08);color:#b1d4c7;border:1px solid rgba(142,183,167,.12)}.source,.factors{color:var(--muted);font-size:.68rem;line-height:1.5}.source{margin:0 0 12px}.market-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.market{min-height:72px;padding:10px;border-radius:13px;background:rgba(2,8,7,.42);border:1px solid rgba(52,211,153,.10)}.market-label{display:block;color:var(--muted2);font-size:.59rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;margin-bottom:7px}.market-value{font-size:.78rem;color:#e7fff6;font-weight:660;line-height:1.42}.factors{border-top:1px solid rgba(52,211,153,.10);margin-top:12px;padding-top:11px}.empty{grid-column:1/-1;border:1px dashed rgba(52,245,164,.22);border-radius:20px;padding:36px 16px;text-align:center;color:var(--muted);background:rgba(7,24,20,.55)}.detail-overlay{position:fixed;inset:0;display:none;z-index:99;background:rgba(0,0,0,.62);backdrop-filter:blur(8px);padding:18px;overflow:auto}.detail-overlay.open{display:block}.detail-panel{max-width:960px;margin:28px auto;padding:18px;border:1px solid rgba(52,211,153,.14);background:rgba(3,17,14,.96);border-radius:18px;box-shadow:0 24px 80px rgba(0,0,0,.50)}.detail-top{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:14px}.detail-title{font-size:1.4rem;font-weight:920}.detail-close{cursor:pointer;border:1px solid rgba(52,211,153,.18);background:rgba(2,8,7,.62);color:var(--text);border-radius:12px;padding:8px 10px;font-weight:800}.detail-sections{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.detail-section{border:1px solid rgba(52,211,153,.12);background:rgba(2,8,7,.42);border-radius:15px;padding:12px}.detail-section h3{margin:0 0 10px;font-size:.78rem;letter-spacing:.10em;text-transform:uppercase;color:var(--green)}.detail-row{display:flex;justify-content:space-between;gap:10px;padding:7px 0;border-bottom:1px solid rgba(52,211,153,.07);color:#c8f7e4;font-size:.72rem}.detail-row:last-child{border-bottom:0}.detail-row span:first-child{color:var(--muted)}.policy{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.policy-card{padding:16px;border-radius:18px;border:1px solid rgba(52,211,153,.12);background:rgba(7,24,20,.62)}.policy-title{font-weight:850;margin-bottom:8px}.policy-copy{color:var(--muted);font-size:.74rem;line-height:1.55;margin:0}.footer{text-align:center;margin-top:30px;color:var(--muted2);font-size:.7rem;line-height:1.6}@media(max-width:1080px){.hero-content{grid-template-columns:1fr}.stats{grid-template-columns:repeat(3,minmax(0,1fr))}.health-grid,.guardrail-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.games{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:680px){.container{padding:12px 11px 24px}.stats,.health-grid,.guardrail-grid,.games,.policy,.detail-sections{grid-template-columns:1fr}.hero{padding:20px 16px;border-radius:21px}.stat-value,.health-value,.guardrail-value{font-size:1.24rem}}
</style>
</head>
<body>
<div class="container">
  <div class="topbar"><div class="brand"><div class="brand-mark">MLB</div><div><div class="brand-name">MLB Intelligence Cloud</div><div class="brand-sub">Emerald Quant Terminal</div></div></div><div class="live-pill"><span class="live-dot"></span> Live locked</div></div>
  <section class="hero"><div class="hero-content"><div><div class="hero-kicker">AI Sports Research Â· MLB only Â· Paper tracking</div><h1><span class="hero-gradient">Multi-game AI analysis with model guardrails</span></h1><p class="hero-copy">Product-style MLB game board with market comparison, signal cases, lineup quality, confidence caps, paper-only tracking, and transparent model governance. No automated wagering. No user funds. No live betting.</p><div id="update-time" class="updated">Loading market update...</div></div><aside class="terminal-card"><div class="terminal-title"><span>Governance Status</span><span class="terminal-status">LOCKED</span></div><div class="terminal-lines"><div class="terminal-line"><span>Mode</span><span>Shadow Research</span></div><div class="terminal-line"><span>Execution</span><span>No Automated Wagering</span></div><div class="terminal-line"><span>Evidence</span><span>CLV Â· OOS Â· Calibration</span></div></div></aside></div></section>
  <div id="messages" class="messages"></div>
  <div class="section-heading"><h2>Mission Control Pulse</h2><span>Clean settled snapshots and market evidence</span></div>
  <section class="stats"><div class="stat"><div class="stat-label">Settled</div><div id="total" class="stat-value neutral">--</div><div class="stat-caption">clean predictions</div></div><div class="stat featured"><div class="stat-label">ML ROI</div><div id="roi" class="stat-value neutral">--</div><div id="roi-caption" class="stat-caption">paper bets</div></div><div class="stat"><div class="stat-label">Win Rate</div><div id="win-rate" class="stat-value neutral">--</div><div class="stat-caption">moneyline bets</div></div><div class="stat"><div class="stat-label">Brier</div><div id="brier" class="stat-value neutral">--</div><div class="stat-caption">probability score</div></div><div class="stat featured"><div class="stat-label">Avg CLV</div><div id="avg-clv" class="stat-value waiting">Waiting</div><div id="clv-caption" class="stat-caption">entry vs closing line</div></div><div class="stat"><div class="stat-label">Positive CLV</div><div id="positive-clv" class="stat-value waiting">--</div><div id="positive-clv-caption" class="stat-caption">price capture rate</div></div></section>
  <div class="section-heading"><h2>Accuracy Diagnostics</h2><span>Breakdown of settled clean predictions</span></div>
  <section class="stats"><div class="stat"><div class="stat-label">All Settled</div><div id="acc-all" class="stat-value neutral">--</div><div id="acc-all-caption" class="stat-caption">all clean predictions</div></div><div class="stat featured"><div class="stat-label">ML Bets</div><div id="acc-ml-bets" class="stat-value neutral">--</div><div id="acc-ml-bets-caption" class="stat-caption">paper bet sample</div></div><div class="stat"><div class="stat-label">Home Picks</div><div id="acc-home" class="stat-value neutral">--</div><div id="acc-home-caption" class="stat-caption">model home picks</div></div><div class="stat"><div class="stat-label">Away Picks</div><div id="acc-away" class="stat-value neutral">--</div><div id="acc-away-caption" class="stat-caption">model away picks</div></div><div class="stat"><div class="stat-label">Favorites</div><div id="acc-favorites" class="stat-value neutral">--</div><div id="acc-favorites-caption" class="stat-caption">market favorite picks</div></div><div class="stat"><div class="stat-label">Underdogs</div><div id="acc-underdogs" class="stat-value neutral">--</div><div id="acc-underdogs-caption" class="stat-caption">market underdog picks</div></div></section>
  <div class="section-heading"><h2>Research Guardrails</h2><span>Sample gates, confidence caps, freshness, and slice policy</span></div>
  <section class="guardrail-grid"><div class="guardrail-card"><div class="guardrail-label">Sample Gate</div><div id="guardrail-sample" class="guardrail-value waiting">--</div><div id="guardrail-sample-caption" class="guardrail-caption">waiting for sample state</div></div><div class="guardrail-card"><div class="guardrail-label">Promotion Gate</div><div id="guardrail-promotion" class="guardrail-value waiting">--</div><div id="guardrail-promotion-caption" class="guardrail-caption">shadow only</div></div><div class="guardrail-card"><div class="guardrail-label">Confidence Cap</div><div id="guardrail-confidence" class="guardrail-value waiting">--</div><div id="guardrail-confidence-caption" class="guardrail-caption">loading confidence guardrail</div></div><div class="guardrail-card"><div class="guardrail-label">Freshness</div><div id="guardrail-freshness" class="guardrail-value waiting">--</div><div id="guardrail-freshness-caption" class="guardrail-caption">source recency</div></div></section>
  <div class="section-heading"><h2>System Health Matrix</h2><span>Pipeline reliability and source coverage</span></div>
  <section class="health-grid"><div class="health-card"><div class="health-label">Status</div><div id="health-status" class="health-value waiting">Loading</div><div id="health-status-caption" class="health-caption">checking pipeline</div></div><div class="health-card"><div class="health-label">Predictions</div><div id="health-predictions" class="health-value neutral">--</div><div id="health-predictions-caption" class="health-caption">today board</div></div><div class="health-card"><div class="health-label">Odds</div><div id="health-odds" class="health-value neutral">--</div><div class="health-caption">OK / suspicious / unavailable</div></div><div class="health-card"><div class="health-label">Context Ready</div><div id="health-context" class="health-value neutral">--</div><div id="health-context-caption" class="health-caption">pregame context</div></div><div class="health-card"><div class="health-label">Snapshots</div><div id="health-snapshots" class="health-value neutral">--</div><div id="health-snapshots-caption" class="health-caption">clean / settled</div></div><div class="health-card"><div class="health-label">Market Rows</div><div id="health-market" class="health-value neutral">--</div><div id="health-market-caption" class="health-caption">closing ML rows</div></div><div class="health-card"><div class="health-label">Lineup Feed</div><div id="health-lineup-feed" class="health-value neutral">--</div><div id="health-lineup-feed-caption" class="health-caption">batting order coverage</div></div><div class="health-card"><div class="health-label">Model Readiness</div><div id="health-training" class="health-value neutral">--</div><div id="health-training-caption" class="health-caption">training status</div></div></section>
  <div class="section-heading"><h2>Signal Center</h2><span>Cases grouped by paper signal, tracking, risk block, market role, and lineup quality</span></div><section class="signal-center"><div id="case-tabs" class="case-tabs"><button class="case-tab active" data-case="all">All</button></div></section>
  <div class="section-heading"><h2>Game Board</h2><span>Product-style single-game analysis cards</span></div><section id="games" class="games"><div class="empty">Loading today's game board...</div></section>
  <section class="policy" style="margin-top:14px"><div class="policy-card"><div class="policy-title">Evidence-first market tracking</div><p class="policy-copy">Paper entries record visible prices at recommendation time. CLV compares entry price with closing market median. Positive CLV is evidence of price capture, not proof of future profitability.</p></div><div class="policy-card"><div class="policy-title">Paper-only governance</div><p class="policy-copy">Live betting remains locked until sample size, rolling OOS validation, calibration, CLV, and risk gates all pass. This interface does not execute wagers.</p></div></section>
  <div id="detail-overlay" class="detail-overlay"><div class="detail-panel"><div class="detail-top"><div><div id="detail-title" class="detail-title">Game Detail</div><div id="detail-subtitle" class="game-time">Loading analysis...</div></div><button class="detail-close" onclick="closeGameDetail()">Close</button></div><div id="detail-body" class="detail-sections"></div></div></div>
  <div class="footer">MLB Intelligence Cloud - Live betting disabled - No automated wagering<br>Research dashboard for market evidence, model quality, data governance, and SaaS readiness.</div>
</div>
<script>
let activeCase="all";
function escapeText(v){return String(v==null?"":v).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#39;")}
function parseUtcTimestamp(raw){if(!raw)return null;const t=String(raw);const n=/(?:Z|[+-]\d{2}:\d{2})$/i.test(t)?t:`${t}Z`;const d=new Date(n);return Number.isNaN(d.valueOf())?null:d}
function formatPercent(v,d=1,s=false){const n=Number(v);if(!Number.isFinite(n))return"--";const sign=s&&n>0?"+":"";return`${sign}${(n*100).toFixed(d)}%`}
function displayTime(row){const raw=row.start_time||row.game_datetime||row.game_time||row.game_date;if(!raw)return"--";if(/^\d{4}-\d{2}-\d{2}$/.test(String(raw)))return"Time pending";const p=parseUtcTimestamp(raw);if(!p)return"--";return p.toLocaleString("zh-TW",{timeZone:"Asia/Taipei",month:"numeric",day:"numeric",hour:"2-digit",minute:"2-digit",hour12:false})}
function signalStatusLabel(s){const v=String(s||"TRACKING_ONLY").toUpperCase();if(v==="PAPER_SIGNAL")return"Paper Signal";if(v==="TRACKING_ONLY")return"Tracking Only";if(v==="BLOCKED_BY_RISK")return"Blocked by Risk";if(v==="BLOCKED_BY_DATA")return"Blocked by Data";if(v==="NO_SIGNAL")return"No Signal";return v.replaceAll("_"," ")}
function productSignalClass(s){const v=String(s||"").toUpperCase();if(v==="PAPER_SIGNAL")return"paper";if(v.startsWith("BLOCKED"))return"blocked";if(v==="NO_SIGNAL")return"no-signal";return"track"}
function caseLabel(id){return{all:"All",paper_signals:"Paper",tracking_only:"Tracking",blocked:"Blocked",positive_edge:"Positive Edge",high_edge:"High Edge",favorites:"Favorites",underdogs:"Underdogs",lineup_missing:"Lineup Missing",high_confidence:"High Confidence",no_signal:"No Signal"}[id]||id.replaceAll("_"," ")}
function renderResearchGuardrails(data){const s=data.summary||{},c=Number(s.clean_settled_samples||0),t=Number(s.minimum_clean_train_samples||300),p=Number(s.minimum_promotion_samples||500),mc=s.recommended_max_display_confidence;document.getElementById("guardrail-sample").textContent=`${c} / ${t}`;document.getElementById("guardrail-sample").className=`guardrail-value ${c>=t?"positive":"waiting"}`;document.getElementById("guardrail-sample-caption").textContent=c>=t?"training sample gate met":"training sample gate not met";document.getElementById("guardrail-promotion").textContent=`${c} / ${p}`;document.getElementById("guardrail-promotion").className=`guardrail-value ${c>=p?"positive":"waiting"}`;document.getElementById("guardrail-promotion-caption").textContent="production replacement locked";document.getElementById("guardrail-confidence").textContent=mc==null?"Capped":formatPercent(mc,0);document.getElementById("guardrail-confidence-caption").textContent=s.block_high_confidence_language?"high-confidence language blocked":"confidence language allowed in shadow only";document.getElementById("guardrail-freshness").textContent=s.freshness_global_grade||"--";const stale=Array.isArray(s.stale_sources)?s.stale_sources:[];document.getElementById("guardrail-freshness-caption").textContent=stale.length?`stale: ${stale.slice(0,3).join(", ")}`:"all tracked sources acceptable"}
function renderSignalTabs(board){const counts=board.case_counts||{},target=document.getElementById("case-tabs"),ordered=["all","paper_signals","tracking_only","blocked","positive_edge","high_edge","favorites","underdogs","lineup_missing","high_confidence","no_signal"];target.innerHTML=ordered.filter(id=>id==="all"||Number(counts[id]||0)>0).map(id=>`<button class="case-tab${id===activeCase?" active":""}" data-case="${escapeText(id)}">${escapeText(caseLabel(id))} ${Number(counts[id]||0)?`(${counts[id]})`:""}</button>`).join("");[...target.querySelectorAll(".case-tab")].forEach(b=>b.addEventListener("click",()=>{activeCase=b.getAttribute("data-case")||"all";renderSignalTabs(board);renderProductGameBoard(board)}))}
function productGameCard(g){const prob=g.selected_probability==null?"--":formatPercent(g.selected_probability),edge=g.edge==null?"Edge --":`Edge ${formatPercent(g.edge,1,true)}`,marketHome=g.market_home_probability==null?"--":formatPercent(g.market_home_probability),homeProb=g.home_probability==null?"--":formatPercent(g.home_probability),lineupGrade=g.lineup_context&&g.lineup_context.lineup_confidence_grade?g.lineup_context.lineup_confidence_grade:"missing",reasons=Array.isArray(g.signal_reasons)?g.signal_reasons:[],features=Array.isArray(g.top_features)?g.top_features:[];return`<article class="game-card product" onclick="openGameDetail('${escapeText(g.game_id)}')"><div class="signal ${productSignalClass(g.signal_status)}"><span>${escapeText(signalStatusLabel(g.signal_status))}</span><span class="signal-edge">${escapeText(edge)}</span></div><div class="game-body"><div class="matchup-row"><div><div class="matchup">${escapeText(g.matchup||"--")}</div><div class="game-time">${escapeText(displayTime(g))}</div></div><div class="probability">${escapeText(prob)}<small>${escapeText(g.selected_team||"Model pick")}</small></div></div><div class="tags"><span class="tag green">Confidence ${escapeText(g.confidence_grade||"D")}</span><span class="tag ${g.market_role==="underdog"?"amber":"green"}">${escapeText(g.market_role||"unknown")}</span><span class="tag ${lineupGrade==="A"||lineupGrade==="B"?"green":lineupGrade==="C"?"amber":"red"}">Lineup ${escapeText(lineupGrade)}</span><span class="tag muted">${escapeText(g.odds_quality_status||"odds unknown")}</span></div><p class="source">Model home: ${escapeText(homeProb)} | Market home: ${escapeText(marketHome)}<br>Source: ${escapeText(g.odds_source||"No verified source")}</p><div class="market-grid"><div class="market"><span class="market-label">Moneyline</span><span class="market-value">${escapeText(g.moneyline_recommendation||"NO BET")}</span></div><div class="market"><span class="market-label">Odds</span><span class="market-value">${escapeText(g.home_team||"Home")} ${g.home_moneyline_odds??"--"}<br>${escapeText(g.away_team||"Away")} ${g.away_moneyline_odds??"--"}</span></div><div class="market"><span class="market-label">Spread</span><span class="market-value">${escapeText(g.spread_recommendation||"NO BET")}</span></div><div class="market"><span class="market-label">Total</span><span class="market-value">${escapeText(g.total_recommendation||"NO BET")}</span></div></div><div class="factors">Signal reason: ${reasons.length?reasons.map(escapeText).slice(0,2).join(" - "):"No signal reason available."}</div><div class="factors">Key factors: ${features.length?features.map(escapeText).slice(0,4).join(" - "):"No additional factors available."}</div></div></article>`}
function renderProductGameBoard(board){const target=document.getElementById("games"),games=Array.isArray(board.games)?board.games:[],filtered=games.filter(g=>activeCase==="all"||(Array.isArray(g.case_tags)&&g.case_tags.includes(activeCase)));if(!filtered.length){target.innerHTML=`<div class="empty">No games in ${escapeText(caseLabel(activeCase))}.</div>`;return}target.innerHTML=filtered.map(productGameCard).join("")}
async function openGameDetail(id){const overlay=document.getElementById("detail-overlay"),title=document.getElementById("detail-title"),subtitle=document.getElementById("detail-subtitle"),body=document.getElementById("detail-body");overlay.classList.add("open");title.textContent="Loading game detail...";subtitle.textContent=id;body.innerHTML="";try{const r=await fetch(`/api/game-detail/${encodeURIComponent(id)}`);if(!r.ok)throw new Error(`Game detail API returned ${r.status}`);const p=await r.json(),g=p.game||{},sections=Array.isArray(p.sections)?p.sections:[];title.textContent=g.matchup||"Game Detail";subtitle.textContent=`${signalStatusLabel(g.signal_status)} Â· ${g.selected_team||"No selected team"}`;body.innerHTML=sections.map(s=>`<div class="detail-section"><h3>${escapeText(s.title||s.id||"Section")}</h3>${(Array.isArray(s.items)?s.items:[]).map(i=>`<div class="detail-row"><span>${escapeText(i.label||"")}</span><strong>${escapeText(formatDetailValue(i.value))}</strong></div>`).join("")}</div>`).join("")}catch(e){title.textContent="Game detail unavailable";subtitle.textContent=e.message;body.innerHTML=`<div class="empty">Unable to load game detail.</div>`}}
function closeGameDetail(){document.getElementById("detail-overlay").classList.remove("open")}
function formatDetailValue(v){if(v==null||v==="")return"--";if(typeof v==="number"){if(v>=0&&v<=1)return formatPercent(v);return v.toFixed(2)}if(typeof v==="boolean")return v?"Yes":"No";return String(v)}
function accuracyText(b){return!b||b.accuracy==null?"--":formatPercent(b.accuracy,1)}function accuracyCaption(b,f){return!b?f:`${b.correct??0} / ${b.sample_count??0} correct`}function setAccuracyValue(id,b){const e=document.getElementById(id);if(!e)return;e.textContent=accuracyText(b);const v=b&&b.accuracy!=null?Number(b.accuracy):null;e.className=v==null?"stat-value waiting":v>=.55?"stat-value positive":v<.5?"stat-value negative":"stat-value neutral"}
function renderAccuracyDiagnostics(d){const b=d.accuracy_breakdown||{};setAccuracyValue("acc-all",b.all_settled);document.getElementById("acc-all-caption").textContent=accuracyCaption(b.all_settled,"all clean predictions");setAccuracyValue("acc-ml-bets",b.moneyline_paper_bets);document.getElementById("acc-ml-bets-caption").textContent=accuracyCaption(b.moneyline_paper_bets,"paper bet sample");setAccuracyValue("acc-home",b.home_model_picks);document.getElementById("acc-home-caption").textContent=accuracyCaption(b.home_model_picks,"model home picks");setAccuracyValue("acc-away",b.away_model_picks);document.getElementById("acc-away-caption").textContent=accuracyCaption(b.away_model_picks,"model away picks");setAccuracyValue("acc-favorites",b.favorites);document.getElementById("acc-favorites-caption").textContent=accuracyCaption(b.favorites,"market favorite picks");setAccuracyValue("acc-underdogs",b.underdogs);document.getElementById("acc-underdogs-caption").textContent=accuracyCaption(b.underdogs,"market underdog picks")}
function renderPerformance(d){document.getElementById("total").textContent=d.total??"--";const roi=document.getElementById("roi");if(d.roi==null){roi.textContent="No bets";roi.className="stat-value waiting"}else{roi.textContent=formatPercent(d.roi,1,true);roi.className=`stat-value ${d.roi>=0?"positive":"negative"}`}document.getElementById("roi-caption").textContent=`${d.moneyline_bets||0} settled ML bets`;const wr=document.getElementById("win-rate");wr.textContent=d.win_rate==null?"--":formatPercent(d.win_rate);wr.className=d.win_rate==null?"stat-value waiting":"stat-value neutral";const br=document.getElementById("brier");br.textContent=d.brier==null?"--":Number(d.brier).toFixed(3);br.className=d.brier==null?"stat-value waiting":"stat-value neutral";const ac=document.getElementById("avg-clv"),pc=document.getElementById("positive-clv");if(d.avg_clv==null){ac.textContent="Waiting";ac.className="stat-value waiting";pc.textContent="--";pc.className="stat-value waiting";document.getElementById("clv-caption").textContent=d.clv_message||"closing lines pending"}else{ac.textContent=formatPercent(d.avg_clv,2,true);ac.className=`stat-value ${d.avg_clv>=0?"positive":"negative"}`;pc.textContent=formatPercent(d.positive_clv_rate);pc.className=`stat-value ${d.positive_clv_rate>=.5?"positive":"negative"}`;document.getElementById("clv-caption").textContent=`${d.clv_samples||0} entry vs close samples`}document.getElementById("positive-clv-caption").textContent=`${d.clv_samples||0} CLV samples`}
function renderHealth(d){const s=String(d.status||"UNKNOWN");document.getElementById("health-status").textContent=s;document.getElementById("health-status").className=`health-value ${s==="OK"?"positive":s==="ERROR"?"negative":"waiting"}`;document.getElementById("health-status-caption").textContent=Array.isArray(d.messages)&&d.messages.length?d.messages[0]:"pipeline checked";document.getElementById("health-predictions").textContent=`${d.prediction_count||0}/${d.scheduled_game_count||0}`;document.getElementById("health-odds").textContent=`${d.odds?.ok||0}/${(d.odds?.ok||0)+(d.odds?.suspicious||0)+(d.odds?.unavailable||0)+(d.odds?.missing||0)}`;document.getElementById("health-context").textContent=`${d.daily_context?.ready_context_count||0}`;document.getElementById("health-context-caption").textContent=`${d.daily_context?.latest_context_count||0} latest context rows`;document.getElementById("health-snapshots").textContent=`${d.snapshots?.clean_rows||0}/${d.snapshots?.stored_rows||0}`;document.getElementById("health-snapshots-caption").textContent=`${d.snapshots?.settled_rows||0} settled`;document.getElementById("health-market").textContent=`${d.market_odds_history?.closing_moneyline_rows||0}`;document.getElementById("health-market-caption").textContent=`${d.market_odds_history?.stored_rows||0} market rows`;document.getElementById("health-lineup-feed").textContent=`${d.daily_context?.lineup_player_count_rows||d.daily_context?.confirmed_lineup_count||0}`;document.getElementById("health-lineup-feed-caption").textContent="lineup-ready context rows";const tr=d.training||{};document.getElementById("health-training").textContent=tr.trained?"Trained":`${tr.sample_count||0}/${tr.minimum_required||0}`;document.getElementById("health-training-caption").textContent=tr.reason||"model readiness"}
async function loadDashboard(){let messages=[];try{const r=await fetch("/api/predictions");if(!r.ok)throw new Error(`Predictions API returned ${r.status}`);const p=await r.json();const g=parseUtcTimestamp(p.generated_at);const u=g?g.toLocaleString("zh-TW",{timeZone:"Asia/Taipei",year:"numeric",month:"numeric",day:"numeric",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}):"--";document.getElementById("update-time").textContent=`Updated in Taipei: ${u}`}catch(e){messages.push(`<div class="message bad">Predictions load failed: ${escapeText(e.message)}</div>`)}try{const r=await fetch("/api/research-guardrails");if(!r.ok)throw new Error(`Research guardrails API returned ${r.status}`);renderResearchGuardrails(await r.json())}catch(e){messages.push(`<div class="message warn">Research guardrails load failed: ${escapeText(e.message)}</div>`)}try{const r=await fetch("/api/game-board");if(!r.ok)throw new Error(`Game board API returned ${r.status}`);const b=await r.json();renderSignalTabs(b);renderProductGameBoard(b)}catch(e){messages.push(`<div class="message warn">Product game board load failed: ${escapeText(e.message)}</div>`);document.getElementById("games").innerHTML='<div class="empty">Product game board unavailable.</div>'}try{const r=await fetch("/api/performance");if(!r.ok)throw new Error(`Performance API returned ${r.status}`);const p=await r.json();renderPerformance(p);renderAccuracyDiagnostics(p)}catch(e){messages.push(`<div class="message bad">Performance load failed: ${escapeText(e.message)}</div>`)}try{const r=await fetch("/api/health");if(!r.ok)throw new Error(`Health API returned ${r.status}`);renderHealth(await r.json())}catch(e){messages.push(`<div class="message bad">Health load failed: ${escapeText(e.message)}</div>`)}if(messages.length)document.getElementById("messages").innerHTML=messages.join("")}
loadDashboard();
</script></body></html>
"""


@app.api_route("/", methods=["GET", "HEAD"])
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(child) for child in value]
    if isinstance(value, tuple):
        return [_sanitize_json_value(child) for child in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value) and not isinstance(value, (str, bool)):
            return None
    except Exception:
        pass
    return value


def _read_csv_safe(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), "missing"
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def _read_json_safe(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, str(exc)
    if not isinstance(payload, dict):
        return {}, "json payload is not an object"
    return payload, None


def _read_prediction_report_safe() -> tuple[dict[str, Any] | None, str | None]:
    if not REPORT_PATH.exists():
        return None, "missing"
    try:
        payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "report is not a JSON object"
    return payload, None


def _normalize_game_id(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return ""
    try:
        parsed = float(text)
        if parsed.is_integer():
            return str(int(parsed))
    except Exception:
        pass
    return text


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "valid", "ok"})


def _enrich_start_times(payload: dict[str, Any]) -> dict[str, Any]:
    return payload


@app.get("/api/predictions")
def get_predictions():
    try:
        with REPORT_PATH.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if isinstance(payload, dict):
            payload = _enrich_start_times(payload)
            payload = _sanitize_json_value(payload)
            return JSONResponse(content=payload)
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Static prediction report read failed: {exc}")

    if generate_predictions is None:
        return JSONResponse({"error": "Prediction module not loaded"}, status_code=503)

    try:
        payload = generate_predictions()
        payload = _enrich_start_times(payload)
        payload = _sanitize_json_value(payload)
        return JSONResponse(content=payload)
    except Exception as exc:
        print(f"Real-time generation failed: {exc}")
        return JSONResponse({"error": f"Real-time generation failed: {exc}"}, status_code=500)


def _recommendation_side(recommendation: str, home_team: str, away_team: str) -> str | None:
    text = str(recommendation or "").strip().lower()
    if not text or text in {"no bet", "pass", "no data"}:
        return None
    home_text = str(home_team or "").strip().lower()
    away_text = str(away_team or "").strip().lower()
    if away_text and away_text in text:
        return "away"
    if home_text and home_text in text:
        return "home"
    if "away" in text:
        return "away"
    if "home" in text:
        return "home"
    return None


def _prepare_clean_dashboard_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "game_id" not in frame.columns:
        return pd.DataFrame()
    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()
    if "pipeline_version" in result.columns:
        preferred = result[result["pipeline_version"].astype(str) == CLEAN_PIPELINE_VERSION].copy()
        if not preferred.empty:
            result = preferred
    if "snapshot_valid" in result.columns:
        result = result[_bool_series(result["snapshot_valid"])].copy()
    leakage_columns = [
        "home_win", "home_score", "away_score", "final_score", "home_final_score",
        "away_final_score", "settled_at", "actual_winner", "actual_result",
        "final_home_score", "final_away_score", "postgame_win_probability",
    ]
    result = result.drop(columns=[column for column in leakage_columns if column in result.columns], errors="ignore")
    if "snapshot_created_at" in result.columns:
        result["snapshot_created_at"] = pd.to_datetime(result["snapshot_created_at"], errors="coerce", utc=True)
        result = result.sort_values("snapshot_created_at")
        result = result.groupby("game_id", as_index=False).tail(1)
    return result.reset_index(drop=True)


def _prepare_finalized_dashboard_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "game_id" not in frame.columns:
        return pd.DataFrame()
    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result = result[result["game_id"] != ""].copy()
    if "home_win" not in result.columns:
        if {"home_score", "away_score"}.issubset(set(result.columns)):
            home_score = pd.to_numeric(result["home_score"], errors="coerce")
            away_score = pd.to_numeric(result["away_score"], errors="coerce")
            result["home_win"] = (home_score > away_score).astype("Int64")
        else:
            return pd.DataFrame()
    result["home_win"] = pd.to_numeric(result["home_win"], errors="coerce")
    result = result[result["home_win"].isin([0, 1])].copy()
    result["home_win"] = result["home_win"].astype(int)
    return result[["game_id", "home_win"]].drop_duplicates("game_id", keep="last")


def _combine_dashboard_finalized_outcomes(finalized_games: pd.DataFrame, snapshot_outcomes: pd.DataFrame) -> pd.DataFrame:
    frames = []
    prepared_finalized = _prepare_finalized_dashboard_outcomes(finalized_games)
    if not prepared_finalized.empty:
        frames.append(prepared_finalized)
    prepared_cache = _prepare_finalized_dashboard_outcomes(snapshot_outcomes)
    if not prepared_cache.empty:
        frames.append(prepared_cache)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["game_id"] = combined["game_id"].apply(_normalize_game_id)
    combined = combined[combined["game_id"] != ""].copy()
    return combined.drop_duplicates("game_id", keep="last").reset_index(drop=True)


def _load_finalized_joined_clean_snapshots() -> tuple[pd.DataFrame, str | None]:
    snapshots, snapshot_error = _read_csv_safe(SNAPSHOT_PATH)
    if snapshot_error is not None:
        return pd.DataFrame(), f"prediction_snapshots unavailable: {snapshot_error}"
    finalized, finalized_error = _read_csv_safe(FINALIZED_GAMES_PATH)
    if finalized_error is not None:
        finalized = pd.DataFrame()
    snapshot_outcomes, snapshot_outcomes_error = _read_csv_safe(FINALIZED_SNAPSHOT_OUTCOMES_PATH)
    if snapshot_outcomes_error is not None:
        snapshot_outcomes = pd.DataFrame()
    if finalized.empty and snapshot_outcomes.empty:
        return pd.DataFrame(), "finalized_games and finalized_snapshot_outcomes unavailable"
    clean_snapshots = _prepare_clean_dashboard_snapshots(snapshots)
    finalized_outcomes = _combine_dashboard_finalized_outcomes(finalized, snapshot_outcomes)
    if clean_snapshots.empty:
        return pd.DataFrame(), "No clean pregame snapshots available"
    if finalized_outcomes.empty:
        return pd.DataFrame(), "No trusted finalized outcomes available"
    joined = clean_snapshots.merge(finalized_outcomes, on="game_id", how="inner")
    if joined.empty:
        return pd.DataFrame(), "No clean snapshots join trusted finalized outcomes by game_id"
    return joined.reset_index(drop=True), None


def _build_accuracy_bucket(frame: pd.DataFrame, pick_column: str = "model_pick_side") -> dict[str, Any]:
    if frame.empty:
        return {"sample_count": 0, "correct": 0, "accuracy": None}
    result = frame.copy()
    result["home_win"] = pd.to_numeric(result["home_win"], errors="coerce")
    result = result[result["home_win"].isin([0, 1])].copy()
    if result.empty:
        return {"sample_count": 0, "correct": 0, "accuracy": None}
    correct = (((result[pick_column] == "home") & (result["home_win"] == 1)) | ((result[pick_column] == "away") & (result["home_win"] == 0)))
    sample_count = int(len(result))
    correct_count = int(correct.sum())
    return {"sample_count": sample_count, "correct": correct_count, "accuracy": float(correct_count / sample_count) if sample_count else None}


def _build_accuracy_breakdown(settled: pd.DataFrame) -> dict[str, Any]:
    if settled.empty:
        return {}
    frame = settled.copy()
    probability_column = None
    for column in ["displayed_home_win_pct", "predicted_home_win_pct", "premarket_model_home_prob"]:
        if column in frame.columns:
            probability_column = column
            break
    if probability_column is not None:
        frame["_model_home_prob"] = pd.to_numeric(frame[probability_column], errors="coerce")
    else:
        frame["_model_home_prob"] = 0.5
    frame["model_pick_side"] = frame["_model_home_prob"].apply(lambda value: "home" if pd.notna(value) and value >= 0.5 else "away")
    market_col = "market_no_vig_home_prob" if "market_no_vig_home_prob" in frame.columns else None
    if market_col:
        frame["_market_home_prob"] = pd.to_numeric(frame[market_col], errors="coerce")
        frame["market_favorite_side"] = frame["_market_home_prob"].apply(lambda value: "home" if pd.notna(value) and value >= 0.5 else "away")
        frame["market_role"] = frame.apply(lambda row: "favorite" if row["model_pick_side"] == row["market_favorite_side"] else "underdog", axis=1)
    else:
        frame["market_role"] = "unknown"
    paper_bets = frame[((frame.get("recommendation_status", pd.Series(dtype=str)).astype(str).str.upper() == "PAPER_BET") & (frame.get("odds_quality_status", pd.Series(dtype=str)).astype(str).str.upper() == "OK"))].copy()
    return {
        "all_settled": _build_accuracy_bucket(frame),
        "moneyline_paper_bets": _build_accuracy_bucket(paper_bets),
        "home_model_picks": _build_accuracy_bucket(frame[frame["model_pick_side"] == "home"]),
        "away_model_picks": _build_accuracy_bucket(frame[frame["model_pick_side"] == "away"]),
        "favorites": _build_accuracy_bucket(frame[frame["market_role"] == "favorite"]),
        "underdogs": _build_accuracy_bucket(frame[frame["market_role"] == "underdog"]),
    }


def _load_closing_moneyline() -> pd.DataFrame:
    market, market_error = _read_csv_safe(MARKET_ODDS_PATH)
    if market_error is not None or market.empty:
        return pd.DataFrame()
    required_columns = {"game_id", "market", "side", "odds", "is_closing_snapshot"}
    if not required_columns.issubset(set(market.columns)):
        return pd.DataFrame()
    result = market.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    result["odds"] = pd.to_numeric(result["odds"], errors="coerce")
    result["is_closing_snapshot"] = _bool_series(result["is_closing_snapshot"])
    if "pipeline_version" in result.columns:
        result = result[result["pipeline_version"].astype(str) == CLEAN_PIPELINE_VERSION].copy()
    result = result[(result["market"].astype(str).str.lower() == "moneyline") & result["is_closing_snapshot"] & result["side"].astype(str).str.lower().isin(["home", "away"]) & (result["odds"] > 1.0)].copy()
    return result


def _moneyline_clv_metrics(clean_snapshots: pd.DataFrame) -> dict[str, Any]:
    result = {"avg_clv": None, "positive_clv_rate": None, "clv_samples": 0, "clv_message": "Waiting for closing lines"}
    if clean_snapshots.empty:
        result["clv_message"] = "No clean snapshots for CLV"
        return result
    closing = _load_closing_moneyline()
    if closing.empty:
        result["clv_message"] = "No closing moneyline rows available"
        return result
    clv_values = []
    for _, row in clean_snapshots.iterrows():
        side = _recommendation_side(str(row.get("moneyline_recommendation") or ""), str(row.get("home_team") or ""), str(row.get("away_team") or ""))
        if side is None:
            continue
        entry_column = "home_moneyline_odds" if side == "home" else "away_moneyline_odds"
        entry_odds = row.get(entry_column)
        try:
            entry_odds = float(entry_odds)
        except Exception:
            continue
        if entry_odds <= 1.0:
            continue
        game_closing = closing[(closing["game_id"] == _normalize_game_id(row.get("game_id"))) & (closing["side"].astype(str).str.lower() == side)]["odds"].dropna()
        if game_closing.empty:
            continue
        closing_odds = float(game_closing.median())
        entry_probability = 1.0 / float(entry_odds)
        closing_probability = 1.0 / closing_odds
        clv_values.append(closing_probability - entry_probability)
    if not clv_values:
        return result
    result["avg_clv"] = float(sum(clv_values) / len(clv_values))
    result["positive_clv_rate"] = float(sum(value > 0 for value in clv_values) / len(clv_values))
    result["clv_samples"] = int(len(clv_values))
    result["clv_message"] = "Entry price compared with closing market median"
    return result


@app.get("/api/performance")
def get_performance() -> dict[str, Any]:
    result: dict[str, Any] = {
        "pipeline_version": CLEAN_PIPELINE_VERSION,
        "clean_sample_count": 0,
        "total": 0,
        "roi": None,
        "win_rate": None,
        "brier": None,
        "moneyline_bets": 0,
        "avg_clv": None,
        "positive_clv_rate": None,
        "clv_samples": 0,
        "clv_message": "Waiting for closing lines",
        "accuracy_breakdown": {},
        "message": "No finalized-joined clean samples yet",
    }
    if not SNAPSHOT_PATH.exists():
        result["message"] = "prediction_snapshots.csv is missing"
        return result
    joined, joined_error = _load_finalized_joined_clean_snapshots()
    try:
        snapshots, snapshot_error = _read_csv_safe(SNAPSHOT_PATH)
        if snapshot_error is None:
            clean_snapshots = _prepare_clean_dashboard_snapshots(snapshots)
            result.update(_moneyline_clv_metrics(clean_snapshots))
    except Exception as exc:
        result["clv_message"] = f"CLV calculation unavailable: {exc}"
    if joined_error is not None:
        result["message"] = joined_error
        return result
    settled = joined.copy()
    settled["home_win"] = pd.to_numeric(settled["home_win"], errors="coerce")
    settled = settled[settled["home_win"].isin([0, 1])].copy()
    if settled.empty:
        result["message"] = "No finalized-joined clean samples after outcome validation"
        return result
    settled["home_win"] = settled["home_win"].astype(int)
    result["clean_sample_count"] = int(len(settled))
    result["total"] = int(len(settled))
    result["accuracy_breakdown"] = _build_accuracy_breakdown(settled)
    if "displayed_home_win_pct" in settled.columns:
        scored = settled[["home_win", "displayed_home_win_pct"]].copy()
        scored["displayed_home_win_pct"] = pd.to_numeric(scored["displayed_home_win_pct"], errors="coerce")
        scored = scored.dropna()
        if not scored.empty:
            result["brier"] = float(brier_score_loss(scored["home_win"], scored["displayed_home_win_pct"]))
    paper_bets = settled[((settled.get("recommendation_status", pd.Series(dtype=str)).astype(str).str.upper() == "PAPER_BET") & (settled.get("odds_quality_status", pd.Series(dtype=str)).astype(str).str.upper() == "OK"))].copy()
    wins: list[int] = []
    profits: list[float] = []
    for _, row in paper_bets.iterrows():
        side = _recommendation_side(row.get("moneyline_recommendation", ""), row.get("home_team", ""), row.get("away_team", ""))
        if side is None:
            continue
        odds_value = row.get("home_moneyline_odds") if side == "home" else row.get("away_moneyline_odds")
        odds = pd.to_numeric(odds_value, errors="coerce")
        if pd.isna(odds) or float(odds) <= 1.0:
            continue
        won = int(row["home_win"] == 1) if side == "home" else int(row["home_win"] == 0)
        wins.append(won)
        profits.append((float(odds) - 1.0) if won else -1.0)
    if profits:
        result["moneyline_bets"] = int(len(profits))
        result["win_rate"] = float(sum(wins) / len(wins))
        result["roi"] = float(sum(profits) / len(profits))
    result["message"] = "Statistics from trusted finalized outcomes joined to clean pregame snapshots"
    return result


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "OK",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prediction_count": 0,
        "scheduled_game_count": 0,
        "odds": {"ok": 0, "suspicious": 0, "unavailable": 0, "missing": 0},
        "daily_context": {
            "file_exists": False,
            "stored_rows": 0,
            "latest_context_count": 0,
            "ready_context_count": 0,
            "game_feed_available_count": 0,
            "starting_pitcher_id_count": 0,
            "lineup_player_count_rows": 0,
            "confirmed_lineup_count": 0,
        },
        "snapshots": {"file_exists": False, "stored_rows": 0, "clean_rows": 0, "settled_rows": 0},
        "market_odds_history": {"file_exists": False, "stored_rows": 0, "closing_moneyline_rows": 0},
        "training": {
            "file_exists": False,
            "trained": False,
            "skipped": False,
            "sample_count": 0,
            "minimum_required": 0,
            "remaining_samples": None,
            "model_type": "",
            "reason": "",
        },
        "messages": [],
    }
    messages: list[str] = result["messages"]
    report, report_error = _read_prediction_report_safe()
    if report is None:
        result["status"] = "WARNING"
        messages.append(f"prediction report unavailable: {report_error}")
    else:
        rows = report.get("today_predictions") or report.get("predictions") or report.get("games") or []
        if isinstance(rows, list):
            result["prediction_count"] = len(rows)
            result["scheduled_game_count"] = int(report.get("scheduled_game_count") or len(rows) or 0)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                quality = str(row.get("odds_quality_status") or "missing").lower()
                if quality == "ok":
                    result["odds"]["ok"] += 1
                elif quality == "suspicious":
                    result["odds"]["suspicious"] += 1
                elif quality == "unavailable":
                    result["odds"]["unavailable"] += 1
                else:
                    result["odds"]["missing"] += 1
    snapshots, snapshot_error = _read_csv_safe(SNAPSHOT_PATH)
    if snapshot_error is None:
        clean = _prepare_clean_dashboard_snapshots(snapshots)
        joined, _ = _load_finalized_joined_clean_snapshots()
        result["snapshots"] = {"file_exists": True, "stored_rows": int(len(snapshots)), "clean_rows": int(len(clean)), "settled_rows": int(len(joined))}
    market, market_error = _read_csv_safe(MARKET_ODDS_PATH)
    if market_error is None:
        closing = _load_closing_moneyline()
        result["market_odds_history"] = {"file_exists": True, "stored_rows": int(len(market)), "closing_moneyline_rows": int(len(closing))}
    context, context_error = _read_csv_safe(DAILY_CONTEXT_PATH)
    if context_error is None:
        result["daily_context"]["file_exists"] = True
        result["daily_context"]["stored_rows"] = int(len(context))
        result["daily_context"]["latest_context_count"] = int(len(context))
        if "context_ready_for_betting" in context.columns:
            result["daily_context"]["ready_context_count"] = int(_bool_series(context["context_ready_for_betting"]).sum())
        if "game_feed_available" in context.columns:
            result["daily_context"]["game_feed_available_count"] = int(_bool_series(context["game_feed_available"]).sum())
        if "lineups_ready" in context.columns:
            result["daily_context"]["confirmed_lineup_count"] = int(_bool_series(context["lineups_ready"]).sum())
        if {"home_lineup_player_count", "away_lineup_player_count"}.issubset(set(context.columns)):
            home_counts = pd.to_numeric(context["home_lineup_player_count"], errors="coerce").fillna(0)
            away_counts = pd.to_numeric(context["away_lineup_player_count"], errors="coerce").fillna(0)
            result["daily_context"]["lineup_player_count_rows"] = int(((home_counts >= 9) & (away_counts >= 9)).sum())
    sample_state, sample_state_error = _read_json_safe(SAMPLE_STATE_PATH)
    training_status, training_error = _read_json_safe(TRAINING_STATUS_PATH)
    if sample_state or training_status:
        result["training"]["file_exists"] = sample_state_error is None or training_error is None
        result["training"]["trained"] = bool(sample_state.get("trained") or training_status.get("trained", False))
        result["training"]["skipped"] = bool(training_status.get("skipped", False))
        result["training"]["sample_count"] = int(sample_state.get("train_eligible_samples") or training_status.get("sample_count") or 0)
        result["training"]["minimum_required"] = int(sample_state.get("minimum_clean_train_samples") or training_status.get("minimum_clean_train_samples") or 0)
        minimum = result["training"]["minimum_required"]
        result["training"]["remaining_samples"] = max(minimum - result["training"]["sample_count"], 0) if minimum else None
        result["training"]["model_type"] = str(training_status.get("model_type", ""))
        result["training"]["reason"] = str(training_status.get("reason", ""))
    if messages and result["status"] == "OK":
        result["status"] = "WARNING"
    return result


@app.get("/api/research-guardrails")
def get_research_guardrails() -> dict[str, Any]:
    report_map = {
        "sample_state": SAMPLE_STATE_PATH,
        "underdog_diagnostic": Path("report/underdog_diagnostic_report.json"),
        "confidence_bucket_guardrail": Path("report/confidence_bucket_guardrail_report.json"),
        "slice_promotion_gate": Path("report/slice_promotion_gate_report.json"),
        "feature_freshness": Path("report/feature_freshness_report.json"),
        "lineup_quality": Path("report/lineup_quality_report.json"),
        "model_correctness": Path("report/model_correctness_report.json"),
        "model_decision_guardrail": Path("report/model_decision_guardrail_report.json"),
        "research_promotion_readiness": Path("report/research_promotion_readiness_report.json"),
    }
    reports: dict[str, Any] = {}
    missing: list[str] = []
    errors: dict[str, str] = {}
    for name, path in report_map.items():
        payload, error = _read_json_safe(path)
        if error is not None:
            reports[name] = {}
            missing.append(name)
            errors[name] = str(error)
        else:
            reports[name] = payload if isinstance(payload, dict) else {}
    sample_state = reports.get("sample_state", {})
    confidence = reports.get("confidence_bucket_guardrail", {})
    slice_gate = reports.get("slice_promotion_gate", {})
    lineup_quality = reports.get("lineup_quality", {})
    freshness = reports.get("feature_freshness", {})
    correctness = reports.get("model_correctness", {})
    confidence_policy = confidence.get("global_policy") if isinstance(confidence.get("global_policy"), dict) else {}
    summary = {
        "clean_settled_samples": int(sample_state.get("clean_settled_snapshots") or 0),
        "train_eligible_samples": int(sample_state.get("train_eligible_samples") or 0),
        "minimum_clean_train_samples": int(sample_state.get("minimum_clean_train_samples") or 300),
        "minimum_promotion_samples": int(sample_state.get("minimum_promotion_samples") or 500),
        "linked_games": int(sample_state.get("linked_games") or 0),
        "link_rate": sample_state.get("link_rate"),
        "shadow_only": True,
        "global_decision": str(slice_gate.get("global_decision") or "NO_PROMOTION_SHADOW_ONLY"),
        "recommended_max_display_confidence": confidence_policy.get("recommended_max_display_confidence"),
        "block_high_confidence_language": bool(confidence_policy.get("block_high_confidence_language", True)),
        "slice_policy": slice_gate.get("paper_entry_policy", {}),
        "lineup_grade_counts": lineup_quality.get("grade_counts", {}),
        "lineup_context_available_count": lineup_quality.get("context_available_count"),
        "freshness_global_grade": freshness.get("global_grade"),
        "stale_sources": freshness.get("stale_sources", []),
        "overall_model_correctness": correctness.get("overall_accuracy"),
        "blocked_filters": correctness.get("blocked_filters", []),
        "recommended_filters": correctness.get("recommended_filters", []),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "partial" if errors else "ok",
        "summary": summary,
        "reports": reports,
        "missing": missing,
        "errors": errors,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }


def _product_rows_from_prediction_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("today_predictions") or report.get("predictions") or report.get("games") or []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        if pd.isna(parsed) or not math.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def _safe_percent_value(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if parsed > 1.0 and parsed <= 100.0:
        return parsed / 100.0
    if parsed < 0.0 or parsed > 1.0:
        return None
    return parsed


def _product_home_probability(prediction: dict[str, Any]) -> float | None:
    for key in ("displayed_home_win_pct", "predicted_home_win_pct", "premarket_model_home_prob", "home_win_probability"):
        value = _safe_percent_value(prediction.get(key))
        if value is not None:
            return value
    return None


def _product_market_home_probability(prediction: dict[str, Any]) -> float | None:
    for key in ("market_no_vig_home_prob", "market_home_prob", "no_vig_home_prob"):
        value = _safe_percent_value(prediction.get(key))
        if value is not None:
            return value
    return None


def _product_selected_side(prediction: dict[str, Any], home_probability: float | None) -> str:
    recommendation_side = _recommendation_side(str(prediction.get("moneyline_recommendation") or ""), str(prediction.get("home_team") or ""), str(prediction.get("away_team") or ""))
    if recommendation_side in {"home", "away"}:
        return recommendation_side
    if home_probability is None:
        return "unknown"
    return "home" if home_probability >= 0.5 else "away"


def _product_selected_probability(prediction: dict[str, Any], home_probability: float | None, selected_side: str) -> float | None:
    if home_probability is None:
        return None
    if selected_side == "home":
        return home_probability
    if selected_side == "away":
        return 1.0 - home_probability
    return max(home_probability, 1.0 - home_probability)


def _product_market_role(selected_side: str, market_home_probability: float | None) -> str:
    if selected_side not in {"home", "away"} or market_home_probability is None:
        return "unknown"
    market_favorite = "home" if market_home_probability >= 0.5 else "away"
    return "favorite" if selected_side == market_favorite else "underdog"


def _product_edge(home_probability: float | None, market_home_probability: float | None, selected_side: str, prediction: dict[str, Any]) -> float | None:
    direct_edge = _safe_float(prediction.get("model_edge_home"))
    if direct_edge is not None:
        return -direct_edge if selected_side == "away" else direct_edge
    if home_probability is None or market_home_probability is None:
        return None
    home_edge = home_probability - market_home_probability
    return -home_edge if selected_side == "away" else home_edge


def _product_confidence_grade(probability: float | None, edge: float | None) -> str:
    if probability is None:
        return "D"
    edge_abs = abs(edge) if edge is not None else 0.0
    if probability >= 0.62 and edge_abs >= 0.04:
        return "A"
    if probability >= 0.58 and edge_abs >= 0.025:
        return "B"
    if probability >= 0.54 or edge_abs >= 0.015:
        return "C"
    return "D"


def _product_signal_status(prediction: dict[str, Any], selected_side: str, market_role: str, confidence_grade: str, guardrail_summary: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    recommendation_status = str(prediction.get("recommendation_status") or "TRACKING_ONLY").strip().upper()
    odds_quality = str(prediction.get("odds_quality_status") or "UNAVAILABLE").strip().upper()
    slice_policy = guardrail_summary.get("slice_policy")
    if not isinstance(slice_policy, dict):
        slice_policy = {}
    if odds_quality in {"SUSPICIOUS", "UNAVAILABLE", "MISSING"}:
        reasons.append(f"odds quality is {odds_quality}")
        return "BLOCKED_BY_DATA", reasons
    if bool(guardrail_summary.get("block_high_confidence_language")) and confidence_grade == "A":
        reasons.append("high-confidence language is capped by guardrail")
        return "TRACKING_ONLY", reasons
    if market_role == "underdog":
        underdog_policy = slice_policy.get("home_underdog") if selected_side == "home" else slice_policy.get("away_underdog")
        if str(underdog_policy).upper() == "PAPER_ENTRY_BLOCKED_BY_RISK":
            reasons.append("underdog slice is blocked by risk guardrail")
            return "BLOCKED_BY_RISK", reasons
    if recommendation_status == "PAPER_BET":
        reasons.append("existing prediction marks this as paper bet")
        return "PAPER_SIGNAL", reasons
    if recommendation_status == "TRACKING_ONLY":
        reasons.append("existing prediction is tracking only")
        return "TRACKING_ONLY", reasons
    if recommendation_status in {"NO_SIGNAL", "NO BET", "PASS"}:
        reasons.append("no actionable model signal")
        return "NO_SIGNAL", reasons
    return "TRACKING_ONLY", reasons


def _product_case_tags(signal_status: str, selected_side: str, market_role: str, edge: float | None, confidence_grade: str, lineup_grade: str) -> list[str]:
    tags = ["all"]
    if signal_status == "PAPER_SIGNAL":
        tags.append("paper_signals")
    if signal_status == "TRACKING_ONLY":
        tags.append("tracking_only")
    if signal_status.startswith("BLOCKED"):
        tags.append("blocked")
    if signal_status == "NO_SIGNAL":
        tags.append("no_signal")
    if edge is not None and edge > 0:
        tags.append("positive_edge")
    if edge is not None and edge >= 0.03:
        tags.append("high_edge")
    if market_role == "favorite":
        tags.append("favorites")
    elif market_role == "underdog":
        tags.append("underdogs")
    if selected_side == "home":
        tags.append("home_picks")
    elif selected_side == "away":
        tags.append("away_picks")
    if confidence_grade in {"A", "B"}:
        tags.append("high_confidence")
    if str(lineup_grade).lower() in {"d", "missing", "", "none", "nan"}:
        tags.append("lineup_missing")
    return sorted(set(str(tag) for tag in tags if tag))


def _load_lineup_quality_lookup() -> dict[str, dict[str, Any]]:
    frame, error = _read_csv_safe(LINEUP_QUALITY_CONTEXT_PATH)
    if error is not None or frame.empty or "game_id" not in frame.columns:
        return {}
    result = frame.copy()
    result["game_id"] = result["game_id"].apply(_normalize_game_id)
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in result.iterrows():
        game_id = _normalize_game_id(row.get("game_id"))
        if not game_id:
            continue
        lookup[game_id] = {
            "lineup_confidence_grade": row.get("lineup_confidence_grade"),
            "lineup_context_available": bool(row.get("lineup_context_available")),
            "home_lineup_quality_score": _safe_float(row.get("home_lineup_quality_score")),
            "away_lineup_quality_score": _safe_float(row.get("away_lineup_quality_score")),
            "lineup_quality_diff": _safe_float(row.get("lineup_quality_diff")),
            "lineup_quality_warning": row.get("lineup_quality_warning"),
        }
    return lookup


def _build_product_game(prediction: dict[str, Any], guardrail_summary: dict[str, Any], lineup_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    game_id = _normalize_game_id(prediction.get("game_id"))
    home_probability = _product_home_probability(prediction)
    market_home_probability = _product_market_home_probability(prediction)
    selected_side = _product_selected_side(prediction, home_probability)
    selected_probability = _product_selected_probability(prediction, home_probability, selected_side)
    market_role = _product_market_role(selected_side, market_home_probability)
    edge = _product_edge(home_probability, market_home_probability, selected_side, prediction)
    confidence_grade = _product_confidence_grade(selected_probability, edge)
    lineup_context = lineup_lookup.get(game_id, {})
    lineup_grade = str(lineup_context.get("lineup_confidence_grade") or "missing")
    signal_status, signal_reasons = _product_signal_status(prediction, selected_side, market_role, confidence_grade, guardrail_summary)
    case_tags = _product_case_tags(signal_status, selected_side, market_role, edge, confidence_grade, lineup_grade)
    away_team = str(prediction.get("away_team") or "Away")
    home_team = str(prediction.get("home_team") or "Home")
    selected_team = home_team if selected_side == "home" else away_team if selected_side == "away" else ""
    top_features = prediction.get("top_features")
    if not isinstance(top_features, list):
        top_features = []
    block_details = prediction.get("recommendation_block_details")
    if not isinstance(block_details, list):
        block_details = []
    game = {
        "game_id": game_id,
        "game_date": prediction.get("game_date"),
        "start_time": prediction.get("start_time") or prediction.get("game_datetime") or prediction.get("game_time"),
        "matchup": f"{away_team} @ {home_team}",
        "away_team": away_team,
        "home_team": home_team,
        "selected_side": selected_side,
        "selected_team": selected_team,
        "selected_probability": selected_probability,
        "home_probability": home_probability,
        "market_home_probability": market_home_probability,
        "market_role": market_role,
        "edge": edge,
        "confidence_grade": confidence_grade,
        "signal_status": signal_status,
        "signal_reasons": signal_reasons,
        "case_tags": case_tags,
        "moneyline_recommendation": prediction.get("moneyline_recommendation"),
        "spread_recommendation": prediction.get("spread_recommendation"),
        "total_recommendation": prediction.get("total_recommendation"),
        "recommendation_status": prediction.get("recommendation_status"),
        "recommendation_block_reason": prediction.get("recommendation_block_reason"),
        "recommendation_block_details": block_details[:5],
        "odds_quality_status": prediction.get("odds_quality_status"),
        "odds_source": prediction.get("odds_source"),
        "home_moneyline_odds": prediction.get("home_moneyline_odds"),
        "away_moneyline_odds": prediction.get("away_moneyline_odds"),
        "spread_line": prediction.get("spread_line"),
        "total_line": prediction.get("total_line"),
        "lineup_context": lineup_context,
        "top_features": top_features[:8],
        "data_quality_status": prediction.get("data_quality_status"),
        "model_governance_status": prediction.get("model_governance_status"),
        "raw_prediction": prediction,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }
    return _sanitize_json_value(game)


def _game_sort_key(game: dict[str, Any]) -> tuple[int, float, float]:
    status_rank = {"PAPER_SIGNAL": 0, "TRACKING_ONLY": 1, "BLOCKED_BY_RISK": 2, "BLOCKED_BY_DATA": 3, "NO_SIGNAL": 4}.get(str(game.get("signal_status")), 5)
    edge = _safe_float(game.get("edge")) or 0.0
    probability = _safe_float(game.get("selected_probability")) or 0.0
    return (status_rank, -abs(edge), -probability)


@app.get("/api/game-board")
def get_game_board() -> dict[str, Any]:
    report, report_error = _read_prediction_report_safe()
    guardrails = get_research_guardrails()
    guardrail_summary = guardrails.get("summary", {}) if isinstance(guardrails, dict) else {}
    if report is None:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "error",
            "error": report_error or "prediction report unavailable",
            "games": [],
            "case_counts": {},
            "live_betting_allowed": False,
            "automated_wagering_allowed": False,
            "production_model_replacement_allowed": False,
        }
    report = _enrich_start_times(report)
    rows = _product_rows_from_prediction_report(report)
    lineup_lookup = _load_lineup_quality_lookup()
    games = [_build_product_game(row, guardrail_summary, lineup_lookup) for row in rows]
    games = sorted(games, key=_game_sort_key)
    case_counts: dict[str, int] = {}
    for game in games:
        for tag in game.get("case_tags", []):
            case_counts[tag] = case_counts.get(tag, 0) + 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prediction_generated_at": report.get("generated_at"),
        "status": "ok",
        "sport": "MLB",
        "league_scope": "mlb_only",
        "game_count": len(games),
        "case_counts": case_counts,
        "guardrail_summary": guardrail_summary,
        "games": games,
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }


@app.get("/api/signal-center")
def get_signal_center() -> dict[str, Any]:
    board = get_game_board()
    games = board.get("games", [])
    if not isinstance(games, list):
        games = []
    case_labels = {
        "all": "All Games",
        "paper_signals": "Paper Signals",
        "tracking_only": "Tracking Only",
        "blocked": "Blocked",
        "positive_edge": "Positive Edge",
        "high_edge": "High Edge",
        "favorites": "Favorites",
        "underdogs": "Underdogs",
        "lineup_missing": "Lineup Missing",
        "high_confidence": "High Confidence",
        "no_signal": "No Signal",
    }
    cases: dict[str, dict[str, Any]] = {}
    for case_id, label in case_labels.items():
        grouped = [game for game in games if case_id in set(game.get("case_tags", []))]
        cases[case_id] = {"case_id": case_id, "label": label, "count": len(grouped), "games": grouped}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": board.get("status", "ok"),
        "cases": cases,
        "default_case": "all",
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }


@app.get("/api/game-detail/{game_id}")
def get_game_detail(game_id: str) -> dict[str, Any]:
    normalized_id = _normalize_game_id(game_id)
    board = get_game_board()
    games = board.get("games", [])
    if not isinstance(games, list):
        games = []
    selected = None
    for game in games:
        if _normalize_game_id(game.get("game_id")) == normalized_id:
            selected = game
            break
    if selected is None:
        raise HTTPException(status_code=404, detail="Game not found")
    raw = selected.get("raw_prediction")
    if not isinstance(raw, dict):
        raw = {}
    data_quality = raw.get("data_quality_status")
    if not isinstance(data_quality, dict):
        data_quality = {}
    governance = raw.get("model_governance_status")
    if not isinstance(governance, dict):
        governance = {}
    lineup_context = selected.get("lineup_context")
    if not isinstance(lineup_context, dict):
        lineup_context = {}
    sections = [
        {"id": "matchup_overview", "title": "Matchup Overview", "items": [
            {"label": "Matchup", "value": selected.get("matchup")},
            {"label": "Start Time", "value": selected.get("start_time") or selected.get("game_date")},
            {"label": "Selected Team", "value": selected.get("selected_team")},
            {"label": "Market Role", "value": selected.get("market_role")},
        ]},
        {"id": "ai_prediction", "title": "AI Prediction", "items": [
            {"label": "Signal Status", "value": selected.get("signal_status")},
            {"label": "Confidence Grade", "value": selected.get("confidence_grade")},
            {"label": "Selected Probability", "value": selected.get("selected_probability")},
            {"label": "Home Probability", "value": selected.get("home_probability")},
        ]},
        {"id": "market_comparison", "title": "Market Comparison", "items": [
            {"label": "Market Home Probability", "value": selected.get("market_home_probability")},
            {"label": "Model Edge", "value": selected.get("edge")},
            {"label": "Home ML Odds", "value": selected.get("home_moneyline_odds")},
            {"label": "Away ML Odds", "value": selected.get("away_moneyline_odds")},
            {"label": "Odds Source", "value": selected.get("odds_source")},
        ]},
        {"id": "lineup_quality", "title": "Lineup Quality", "items": [
            {"label": "Lineup Grade", "value": lineup_context.get("lineup_confidence_grade")},
            {"label": "Home Score", "value": lineup_context.get("home_lineup_quality_score")},
            {"label": "Away Score", "value": lineup_context.get("away_lineup_quality_score")},
            {"label": "Warning", "value": lineup_context.get("lineup_quality_warning")},
        ]},
        {"id": "risk_guardrail", "title": "Risk & Guardrail", "items": [
            {"label": "Recommendation Status", "value": selected.get("recommendation_status")},
            {"label": "Odds Quality", "value": selected.get("odds_quality_status")},
            {"label": "Block Reason", "value": selected.get("recommendation_block_reason")},
            {"label": "Live Betting Allowed", "value": False},
            {"label": "Automated Wagering Allowed", "value": False},
        ]},
        {"id": "model_governance", "title": "Model Governance", "items": [
            {"label": "Mode", "value": governance.get("mode")},
            {"label": "Clean Model Samples", "value": governance.get("clean_model_sample_count")},
            {"label": "Minimum Samples", "value": governance.get("min_clean_train_samples")},
            {"label": "Prediction Allowed", "value": data_quality.get("prediction_allowed")},
            {"label": "Bet Allowed", "value": data_quality.get("bet_allowed")},
        ]},
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ok",
        "game": selected,
        "sections": sections,
        "top_features": selected.get("top_features", []),
        "signal_reasons": selected.get("signal_reasons", []),
        "case_tags": selected.get("case_tags", []),
        "live_betting_allowed": False,
        "automated_wagering_allowed": False,
        "production_model_replacement_allowed": False,
    }


@app.post("/run")
def run_background(authorization: str = Header(None)) -> dict[str, str]:
    if not ADMIN_TOKEN or not authorization or authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    def task() -> None:
        try:
            if generate_predictions is not None:
                generate_predictions()
        except Exception as exc:
            print(f"Background prediction error: {exc}")
    threading.Thread(target=task, daemon=True).start()
    return {"status": "started"}


@app.post("/train-nrfi")
def train_nrfi(authorization: str = Header(None)) -> dict[str, str]:
    if not ADMIN_TOKEN or not authorization or authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": "disabled", "message": "NRFI training is currently disabled"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
