# main.py
"""Premium FastAPI dashboard for MLB full-game market analytics.

This version keeps the existing prediction report contract and adds:
- Premium dark mobile-first dashboard layout.
- Explicit Taipei-time rendering for report timestamps and game starts.
- Full-game Moneyline, Spread and Total presentation only.
- Market integrity and bookmaker source visibility.
- Clean forward-snapshot performance metrics.
- Moneyline CLV metrics derived from entry prices versus closing prices.

Source text is intentionally ASCII-only so browser-based edits stay safe.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sklearn.metrics import brier_score_loss

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ADMIN_TOKEN = os.getenv("ADMIN_API_TOKEN")
if not ADMIN_TOKEN:
    print("FATAL: ADMIN_API_TOKEN environment variable not set. Exiting.")
    sys.exit(1)

try:
    from prediction import generate_predictions
except Exception as exc:
    print(f"Warning: prediction module failed to import: {exc}")
    generate_predictions = None

app = FastAPI(title="MLB Prediction Hub")

REPORT_PATH = Path("report/prediction.json")
HISTORY_PATH = Path("data/historical_predictions.csv")
SNAPSHOT_PATH = Path("data/prediction_snapshots.csv")
MARKET_ODDS_PATH = Path("data/market_odds_history.csv")
CLEAN_PIPELINE_VERSION = "baseline_v2_clean"


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>MLB Prediction Hub</title>
<style>
:root {
  --bg:#070816;
  --bg-2:#0b1022;
  --panel:#10162e;
  --panel-2:#141b38;
  --panel-3:#181f42;
  --border:rgba(129,140,248,.16);
  --border-strong:rgba(192,132,252,.34);
  --text:#f5f7ff;
  --muted:#8f9abd;
  --muted-2:#667197;
  --green:#35ef9d;
  --green-bg:rgba(53,239,157,.13);
  --pink:#ff62ba;
  --purple:#8e48ff;
  --blue:#53c8ff;
  --amber:#ffc55e;
  --amber-bg:rgba(255,197,94,.13);
  --red:#ff6577;
  --red-bg:rgba(255,101,119,.13);
  --shadow:0 22px 48px rgba(0,0,0,.28);
  --gradient:linear-gradient(92deg,#7d3eff 0%,#cf47dc 47%,#ff62ae 100%);
  --gradient-soft:linear-gradient(140deg,rgba(125,62,255,.26),rgba(255,98,174,.08));
}

* { box-sizing:border-box; }

html, body {
  margin:0;
  min-height:100%;
  background:var(--bg);
  color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Arial,sans-serif;
}

body {
  background:
    radial-gradient(circle at 10% -4%, rgba(126,63,255,.24), transparent 33%),
    radial-gradient(circle at 92% 0%, rgba(255,98,174,.13), transparent 30%),
    linear-gradient(180deg, #070816 0%, #090c1b 48%, #070816 100%);
}

a { color:inherit; }

.container {
  max-width:1320px;
  margin:0 auto;
  padding:20px 18px 36px;
}

.topbar {
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:14px;
  margin-bottom:20px;
}

.brand {
  display:flex;
  align-items:center;
  gap:12px;
}

.brand-mark {
  display:grid;
  place-items:center;
  width:43px;
  height:43px;
  border-radius:14px;
  background:var(--gradient);
  box-shadow:0 9px 22px rgba(160,64,255,.34);
  color:white;
  font-weight:900;
  letter-spacing:-.05em;
  font-size:1.02rem;
}

.brand-name {
  font-size:1.02rem;
  font-weight:700;
  letter-spacing:.01em;
}

.brand-sub {
  color:var(--muted);
  font-size:.69rem;
  letter-spacing:.16em;
  margin-top:3px;
  text-transform:uppercase;
}

.live-pill {
  display:flex;
  align-items:center;
  gap:7px;
  border:1px solid rgba(53,239,157,.24);
  background:rgba(53,239,157,.08);
  color:var(--green);
  border-radius:999px;
  padding:8px 11px;
  font-size:.72rem;
  font-weight:700;
  white-space:nowrap;
}

.live-dot {
  width:7px;
  height:7px;
  border-radius:50%;
  background:var(--green);
  box-shadow:0 0 12px var(--green);
}

.hero {
  position:relative;
  overflow:hidden;
  padding:23px 22px;
  margin-bottom:16px;
  border-radius:24px;
  background:
    linear-gradient(125deg, rgba(23,28,61,.98), rgba(13,17,38,.98)),
    var(--panel);
  border:1px solid var(--border);
  box-shadow:var(--shadow);
}

.hero::after {
  content:"";
  position:absolute;
  right:-75px;
  top:-95px;
  width:230px;
  height:230px;
  border-radius:50%;
  background:radial-gradient(circle, rgba(207,71,220,.27), transparent 68%);
}

.hero-kicker {
  position:relative;
  z-index:1;
  display:inline-flex;
  gap:7px;
  align-items:center;
  margin-bottom:12px;
  color:#d1b4ff;
  font-size:.69rem;
  font-weight:700;
  letter-spacing:.18em;
  text-transform:uppercase;
}

.hero h1 {
  position:relative;
  z-index:1;
  margin:0;
  font-size:clamp(1.7rem,4vw,2.65rem);
  font-weight:760;
  letter-spacing:-.055em;
  line-height:1.06;
}

.hero-gradient {
  background:linear-gradient(92deg,#f7f3ff 5%,#e2caff 38%,#ff7cbc 96%);
  background-clip:text;
  -webkit-background-clip:text;
  color:transparent;
}

.hero-copy {
  position:relative;
  z-index:1;
  color:var(--muted);
  max-width:670px;
  margin:13px 0 0;
  line-height:1.52;
  font-size:.9rem;
}

.updated {
  position:relative;
  z-index:1;
  display:inline-flex;
  align-items:center;
  gap:7px;
  margin-top:17px;
  padding:8px 11px;
  border-radius:999px;
  border:1px solid rgba(129,140,248,.16);
  background:rgba(255,255,255,.035);
  color:#aab4d5;
  font-size:.73rem;
}

.messages {
  margin:14px 0;
}

.message {
  border-radius:14px;
  padding:12px 14px;
  margin-bottom:9px;
  font-size:.82rem;
  line-height:1.45;
}

.message.warn {
  color:#ffd98e;
  border:1px solid rgba(255,197,94,.24);
  background:var(--amber-bg);
}

.message.bad {
  color:#ff9eaa;
  border:1px solid rgba(255,101,119,.26);
  background:var(--red-bg);
}

.section-heading {
  display:flex;
  align-items:end;
  justify-content:space-between;
  gap:12px;
  margin:22px 2px 12px;
}

.section-heading h2 {
  font-size:1.05rem;
  margin:0;
  letter-spacing:-.02em;
}

.section-heading span {
  color:var(--muted);
  font-size:.72rem;
}

.stats {
  display:grid;
  grid-template-columns:repeat(6, minmax(0, 1fr));
  gap:10px;
}

.stat {
  min-height:104px;
  padding:14px 14px 12px;
  border-radius:17px;
  background:linear-gradient(150deg, rgba(20,27,56,.96), rgba(13,17,38,.98));
  border:1px solid var(--border);
}

.stat.featured {
  background:
    linear-gradient(145deg, rgba(53,239,157,.12), rgba(13,17,38,.96));
  border-color:rgba(53,239,157,.23);
}

.stat-label {
  color:var(--muted);
  font-size:.66rem;
  font-weight:700;
  letter-spacing:.12em;
  text-transform:uppercase;
}

.stat-value {
  margin-top:9px;
  font-size:1.55rem;
  line-height:1;
  font-weight:750;
  letter-spacing:-.04em;
}

.stat-caption {
  margin-top:8px;
  font-size:.69rem;
  color:var(--muted-2);
}

.positive { color:var(--green); }
.negative { color:var(--red); }
.neutral { color:var(--text); }
.waiting { color:var(--amber); font-size:1rem; letter-spacing:0; }

.policy {
  display:grid;
  grid-template-columns:1.35fr .95fr;
  gap:10px;
  margin-top:14px;
}

.policy-card {
  border-radius:17px;
  border:1px solid var(--border);
  background:var(--panel);
  padding:14px;
}

.policy-title {
  display:flex;
  gap:8px;
  align-items:center;
  color:#dde3ff;
  font-size:.82rem;
  font-weight:680;
  margin-bottom:8px;
}

.policy-icon {
  display:grid;
  place-items:center;
  width:22px;
  height:22px;
  border-radius:7px;
  font-size:.72rem;
  color:var(--green);
  background:var(--green-bg);
}

.policy-copy {
  margin:0;
  color:var(--muted);
  line-height:1.5;
  font-size:.75rem;
}

.policy-tags {
  display:flex;
  flex-wrap:wrap;
  gap:7px;
  margin-top:11px;
}

.small-pill {
  padding:5px 8px;
  border-radius:999px;
  color:#aab6df;
  background:rgba(129,140,248,.08);
  border:1px solid rgba(129,140,248,.14);
  font-size:.66rem;
  font-weight:650;
}

.games {
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:13px;
}

.game-card {
  position:relative;
  overflow:hidden;
  border-radius:20px;
  background:linear-gradient(150deg,var(--panel-2),var(--panel));
  border:1px solid var(--border);
  box-shadow:0 14px 32px rgba(0,0,0,.17);
}

.game-card.suspicious {
  border-color:rgba(255,101,119,.33);
}

.signal {
  min-height:42px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:9px;
  padding:10px 13px;
  background:rgba(129,140,248,.08);
  color:#c4cce8;
  font-size:.72rem;
  font-weight:740;
  text-transform:uppercase;
  letter-spacing:.08em;
}

.signal.bet {
  background:var(--gradient);
  color:#fff;
}

.signal.track {
  background:linear-gradient(92deg, rgba(101,110,166,.28), rgba(32,38,72,.28));
}

.signal.bad {
  background:linear-gradient(92deg, rgba(255,101,119,.32), rgba(32,38,72,.28));
  color:#ffc5cd;
}

.signal-edge {
  font-size:.7rem;
  letter-spacing:.02em;
  opacity:.96;
  white-space:nowrap;
}

.game-body {
  padding:14px;
}

.matchup-row {
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
  gap:12px;
  margin-bottom:13px;
}

.matchup {
  font-size:1.03rem;
  font-weight:720;
  letter-spacing:-.02em;
}

.game-time {
  color:var(--muted);
  font-size:.72rem;
  margin-top:5px;
}

.probability {
  text-align:right;
  font-weight:760;
  font-size:1.4rem;
  line-height:1;
  color:#fff;
}

.probability small {
  display:block;
  font-size:.59rem;
  letter-spacing:.09em;
  color:var(--muted);
  margin-top:5px;
  text-transform:uppercase;
}

.tags {
  display:flex;
  flex-wrap:wrap;
  gap:6px;
  margin-bottom:12px;
}

.tag {
  display:inline-flex;
  align-items:center;
  padding:5px 8px;
  border-radius:999px;
  font-size:.63rem;
  line-height:1;
  font-weight:720;
  letter-spacing:.06em;
  text-transform:uppercase;
}

.tag.green {
  background:var(--green-bg);
  color:var(--green);
}

.tag.amber {
  background:var(--amber-bg);
  color:var(--amber);
}

.tag.red {
  background:var(--red-bg);
  color:var(--red);
}

.tag.muted {
  background:rgba(129,140,248,.09);
  color:#a7b4df;
}

.source {
  color:var(--muted);
  font-size:.68rem;
  line-height:1.42;
  margin:0 0 12px;
}

.market-grid {
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:8px;
}

.market {
  min-height:70px;
  padding:9px;
  border-radius:11px;
  background:rgba(6,9,22,.38);
  border:1px solid rgba(129,140,248,.08);
}

.market-label {
  display:block;
  color:var(--muted-2);
  font-size:.6rem;
  font-weight:700;
  letter-spacing:.11em;
  text-transform:uppercase;
  margin-bottom:7px;
}

.market-value {
  font-size:.78rem;
  color:#e7ebff;
  font-weight:620;
  line-height:1.42;
}

.rec {
  display:inline-block;
  border-radius:7px;
  padding:5px 8px;
  font-size:.69rem;
  font-weight:720;
}

.rec.bet {
  color:#042116;
  background:var(--green);
}

.rec.pass {
  color:#aab5d8;
  background:rgba(129,140,248,.12);
}

.rec.no-data {
  color:var(--amber);
  background:var(--amber-bg);
}

.subtext {
  display:block;
  color:var(--muted);
  font-size:.63rem;
  margin-top:5px;
}

.factors {
  border-top:1px solid rgba(129,140,248,.1);
  margin-top:12px;
  padding-top:11px;
  color:var(--muted);
  font-size:.68rem;
  line-height:1.5;
}

.empty {
  grid-column:1/-1;
  border:1px dashed rgba(129,140,248,.2);
  border-radius:18px;
  padding:34px 16px;
  text-align:center;
  color:var(--muted);
  background:rgba(16,22,46,.55);
}

.footer {
  text-align:center;
  margin-top:28px;
  color:var(--muted-2);
  font-size:.7rem;
  line-height:1.55;
}

@media (max-width: 1080px) {
  .stats { grid-template-columns:repeat(3,minmax(0,1fr)); }
  .games { grid-template-columns:repeat(2,minmax(0,1fr)); }
}

@media (max-width: 680px) {
  .container {
    padding:12px 11px calc(24px + env(safe-area-inset-bottom));
  }

  .topbar {
    margin-bottom:13px;
  }

  .brand-mark {
    width:39px;
    height:39px;
    border-radius:12px;
  }

  .hero {
    padding:19px 16px;
    border-radius:19px;
  }

  .hero-copy {
    font-size:.78rem;
  }

  .stats {
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:8px;
  }

  .stat {
    min-height:91px;
    padding:12px 11px 10px;
    border-radius:14px;
  }

  .stat-value {
    font-size:1.24rem;
  }

  .stat-caption {
    font-size:.62rem;
  }

  .policy {
    grid-template-columns:1fr;
    gap:8px;
  }

  .games {
    grid-template-columns:1fr;
    gap:11px;
  }

  .game-card {
    border-radius:17px;
  }

  .game-body {
    padding:12px;
  }
}
</style>
</head>
<body>
<div class="container">
  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">MLB</div>
      <div>
        <div class="brand-name">Prediction Hub</div>
        <div class="brand-sub">Full Game Analytics</div>
      </div>
    </div>
    <div class="live-pill"><span class="live-dot"></span> Paper mode</div>
  </div>

  <section class="hero">
    <div class="hero-kicker">Premium market intelligence</div>
    <h1 class="hero-gradient">Full-game MLB value tracker</h1>
    <p class="hero-copy">
      Clean forward testing, verified market quotes and closing-line tracking
      for Moneyline, Spread and Total markets.
    </p>
    <div id="update-time" class="updated">Loading market update...</div>
  </section>

  <div id="messages" class="messages"></div>

  <div class="section-heading">
    <h2>Performance pulse</h2>
    <span>Clean settled snapshots only</span>
  </div>

  <section class="stats">
    <div class="stat">
      <div class="stat-label">Settled</div>
      <div id="total" class="stat-value neutral">--</div>
      <div class="stat-caption">clean predictions</div>
    </div>
    <div class="stat featured">
      <div class="stat-label">ML ROI</div>
      <div id="roi" class="stat-value neutral">--</div>
      <div id="roi-caption" class="stat-caption">paper bets</div>
    </div>
    <div class="stat">
      <div class="stat-label">Win Rate</div>
      <div id="win-rate" class="stat-value neutral">--</div>
      <div class="stat-caption">moneyline bets</div>
    </div>
    <div class="stat">
      <div class="stat-label">Brier</div>
      <div id="brier" class="stat-value neutral">--</div>
      <div class="stat-caption">probability score</div>
    </div>
    <div class="stat featured">
      <div class="stat-label">Avg CLV</div>
      <div id="avg-clv" class="stat-value waiting">Waiting</div>
      <div id="clv-caption" class="stat-caption">closing line pending</div>
    </div>
    <div class="stat">
      <div class="stat-label">Positive CLV</div>
      <div id="positive-clv" class="stat-value waiting">--</div>
      <div id="positive-clv-caption" class="stat-caption">entry vs close</div>
    </div>
  </section>

  <section class="policy">
    <div class="policy-card">
      <div class="policy-title">
        <span class="policy-icon">+</span>
        Transparent market tracking
      </div>
      <p class="policy-copy">
        Paper Bet records the price visible when the recommendation was created.
        CLV compares that entry price against the final pregame market price.
        Positive CLV is evidence of good price capture, not a guarantee of profit.
      </p>
      <div class="policy-tags">
        <span class="small-pill">Moneyline</span>
        <span class="small-pill">Spread</span>
        <span class="small-pill">Total</span>
        <span class="small-pill">No first five markets</span>
      </div>
    </div>
    <div class="policy-card">
      <div class="policy-title">
        <span class="policy-icon">i</span>
        Current model policy
      </div>
      <p class="policy-copy">
        Odds must pass integrity checks before recommendations are labelled as
        paper bets. The next v3 step adds a minimum model edge gate.
      </p>
      <div class="policy-tags">
        <span class="small-pill">baseline_v2_clean</span>
        <span class="small-pill">Forward tested</span>
      </div>
    </div>
  </section>

  <div class="section-heading">
    <h2>Today's full-game board</h2>
    <span>Verified odds and model probabilities</span>
  </div>

  <section id="games" class="games">
    <div class="empty">Loading today's markets...</div>
  </section>

  <div class="footer">
    Data updated hourly - Paper trading only - Full-game markets only<br>
    Past performance and positive CLV do not guarantee future returns.
  </div>
</div>

<script>
function escapeText(value) {
  return String(value == null ? "" : value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function parseUtcTimestamp(raw) {
  if (!raw) return null;
  const text = String(raw);
  const hasTimezone = /(?:Z|[+-]\d{2}:\d{2})$/i.test(text);
  const normalized = hasTimezone ? text : `${text}Z`;
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.valueOf()) ? null : parsed;
}

function formatPercent(value, decimals = 1, signed = false) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  const sign = signed && number > 0 ? "+" : "";
  return `${sign}${(number * 100).toFixed(decimals)}%`;
}

function displayTime(prediction) {
  const raw = prediction.start_time || prediction.game_datetime ||
    prediction.game_time || prediction.game_date;
  if (!raw) return "--";
  if (/^\d{4}-\d{2}-\d{2}$/.test(String(raw))) return "Time pending";

  const parsed = parseUtcTimestamp(raw);
  if (!parsed) return "--";

  return parsed.toLocaleString("zh-TW", {
    timeZone: "Asia/Taipei",
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  });
}

function qualityValue(prediction) {
  return String(prediction.odds_quality_status || "UNAVAILABLE").toUpperCase();
}

function statusValue(prediction) {
  return String(prediction.recommendation_status || "TRACKING_ONLY").toUpperCase();
}

function moneylineRecommendation(prediction) {
  return String(prediction.moneyline_recommendation || "NO BET");
}

function isMoneylineBet(prediction) {
  const recommendation = moneylineRecommendation(prediction).toUpperCase();
  return recommendation !== "NO BET" &&
    recommendation !== "NO DATA" &&
    recommendation !== "PASS";
}

function recommendationSide(prediction) {
  const recommendation = moneylineRecommendation(prediction).toLowerCase();
  const home = String(prediction.home_team || "").toLowerCase();
  const away = String(prediction.away_team || "").toLowerCase();

  if (home && recommendation.includes(home)) return "home";
  if (away && recommendation.includes(away)) return "away";
  return null;
}

function selectedEdge(prediction) {
  const edge = Number(prediction.model_edge_home);
  if (!Number.isFinite(edge)) return null;
  const side = recommendationSide(prediction);
  if (side === "home") return edge;
  if (side === "away") return -edge;
  return edge;
}

function signedLine(value) {
  if (value == null || value === "") return null;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return `${numeric > 0 ? "+" : ""}${numeric.toFixed(1)}`;
}

function badge(value) {
  const label = value || "--";
  if (!value || value === "NO BET" || value === "PASS") {
    return `<span class="rec pass">${escapeText(label)}</span>`;
  }
  if (value === "NO DATA") {
    return `<span class="rec no-data">NO DATA</span>`;
  }
  return `<span class="rec bet">${escapeText(label)}</span>`;
}

function displayOdds(prediction) {
  const home = prediction.home_moneyline_odds ?? prediction.home_odds;
  const away = prediction.away_moneyline_odds ?? prediction.away_odds;
  if (home == null && away == null) return "--";

  const homeText = home == null ? "--" : Number(home).toFixed(2);
  const awayText = away == null ? "--" : Number(away).toFixed(2);

  return `${escapeText(prediction.home_team || "Home")} ${homeText}<br>` +
    `${escapeText(prediction.away_team || "Away")} ${awayText}`;
}

function displayMoneyline(prediction) {
  const recommendation = moneylineRecommendation(prediction);
  if (!isMoneylineBet(prediction)) return badge(recommendation);
  return `${badge(recommendation)}<span class="subtext">Winner only</span>`;
}

function displaySpread(prediction) {
  const recommendation = String(prediction.spread_recommendation || "NO BET");
  if (recommendation === "NO BET" || recommendation === "NO DATA") {
    return badge(recommendation);
  }

  const homeLine = signedLine(prediction.spread_line);
  if (homeLine == null) {
    return `${badge(recommendation)}<span class="subtext">Line unavailable</span>`;
  }

  const recommendationText = recommendation.toLowerCase();
  const homeText = String(prediction.home_team || "").toLowerCase();
  const isHome = homeText && recommendationText.includes(homeText);
  const line = isHome ? homeLine : signedLine(-Number(prediction.spread_line));
  const team = recommendation.replace(/\s+spread$/i, "");

  return badge(`${team} ${line}`);
}

function displayTotal(prediction) {
  const recommendation = String(prediction.total_recommendation || "NO BET");
  if (recommendation === "NO BET" || recommendation === "NO DATA") {
    return badge(recommendation);
  }

  const line = Number(prediction.total_line);
  if (!Number.isFinite(line)) {
    return `${badge(recommendation)}<span class="subtext">Line unavailable</span>`;
  }

  return badge(`${recommendation} ${line.toFixed(1)}`);
}

function keyFactors(prediction) {
  if (Array.isArray(prediction.top_features) && prediction.top_features.length) {
    return prediction.top_features.map(escapeText).join(" - ");
  }

  const features = prediction.features || {};
  return Object.entries(features)
    .filter(([_, value]) => typeof value === "number" && Math.abs(value) > 0.0001)
    .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1]))
    .slice(0, 3)
    .map(([key, value]) => `${escapeText(key)}=${Number(value).toFixed(2)}`)
    .join(" - ");
}

function tagHtml(prediction) {
  const quality = qualityValue(prediction);
  const status = statusValue(prediction);
  const values = [];

  if (status === "PAPER_BET") {
    values.push('<span class="tag green">Paper bet</span>');
  } else {
    values.push('<span class="tag amber">Tracking only</span>');
  }

  if (quality === "OK") {
    values.push('<span class="tag green">Odds ok</span>');
  } else if (quality === "SUSPICIOUS") {
    values.push('<span class="tag red">Odds suspicious</span>');
  } else {
    values.push('<span class="tag amber">Odds unavailable</span>');
  }

  values.push('<span class="tag muted">Full game</span>');
  return values.join("");
}

function signalClass(prediction) {
  const quality = qualityValue(prediction);
  if (quality === "SUSPICIOUS") return "bad";
  if (statusValue(prediction) === "PAPER_BET" && isMoneylineBet(prediction)) {
    return "bet";
  }
  return "track";
}

function signalLabel(prediction) {
  if (qualityValue(prediction) === "SUSPICIOUS") return "Market warning";
  if (statusValue(prediction) === "PAPER_BET" && isMoneylineBet(prediction)) {
    return moneylineRecommendation(prediction);
  }
  return "Tracking opportunity";
}

function renderPerformance(data) {
  document.getElementById("total").textContent = data.total ?? "--";

  const roi = document.getElementById("roi");
  if (data.roi == null) {
    roi.textContent = "No bets";
    roi.className = "stat-value waiting";
  } else {
    roi.textContent = formatPercent(data.roi, 1, true);
    roi.className = `stat-value ${data.roi >= 0 ? "positive" : "negative"}`;
  }

  document.getElementById("roi-caption").textContent =
    `${data.moneyline_bets || 0} settled ML bets`;

  const winRate = document.getElementById("win-rate");
  if (data.win_rate == null) {
    winRate.textContent = "--";
    winRate.className = "stat-value waiting";
  } else {
    winRate.textContent = formatPercent(data.win_rate);
    winRate.className = "stat-value neutral";
  }

  const brier = document.getElementById("brier");
  if (data.brier == null) {
    brier.textContent = "--";
    brier.className = "stat-value waiting";
  } else {
    brier.textContent = Number(data.brier).toFixed(3);
    brier.className = "stat-value neutral";
  }

  const avgClv = document.getElementById("avg-clv");
  const positiveClv = document.getElementById("positive-clv");

  if (data.avg_clv == null) {
    avgClv.textContent = "Waiting";
    avgClv.className = "stat-value waiting";
    positiveClv.textContent = "--";
    positiveClv.className = "stat-value waiting";
    document.getElementById("clv-caption").textContent =
      data.clv_message || "closing lines pending";
  } else {
    avgClv.textContent = formatPercent(data.avg_clv, 2, true);
    avgClv.className = `stat-value ${data.avg_clv >= 0 ? "positive" : "negative"}`;
    positiveClv.textContent = formatPercent(data.positive_clv_rate);
    positiveClv.className = `stat-value ${data.positive_clv_rate >= 0.5 ? "positive" : "negative"}`;
    document.getElementById("clv-caption").textContent =
      `${data.clv_samples || 0} entry vs close samples`;
  }

  document.getElementById("positive-clv-caption").textContent =
    `${data.clv_samples || 0} CLV samples`;
}

function renderGames(data) {
  const rows = data.today_predictions || [];
  const target = document.getElementById("games");
  const messages = [];

  if (!rows.length) {
    target.innerHTML = '<div class="empty">No game data available today.</div>';
    return;
  }

  const suspiciousCount = rows.filter(row => qualityValue(row) === "SUSPICIOUS").length;
  const trackingCount = rows.filter(row => statusValue(row) === "TRACKING_ONLY").length;
  const errors = data.errors || [];

  if (suspiciousCount) {
    messages.push(
      `<div class="message bad">${suspiciousCount} market(s) have suspicious odds and remain tracking only.</div>`
    );
  }

  if (trackingCount) {
    messages.push(
      `<div class="message warn">${trackingCount} market(s) are tracking only because trusted pricing or recommendation requirements were not met.</div>`
    );
  }

  if (errors.length) {
    messages.push(
      `<div class="message warn">The latest report contains ${errors.length} data-quality or feature warning(s).</div>`
    );
  }

  document.getElementById("messages").innerHTML = messages.join("");

  target.innerHTML = rows.map(prediction => {
    const homeProb = prediction.displayed_home_win_pct ??
      prediction.predicted_home_win_pct;
    const edge = selectedEdge(prediction);
    const edgeText = edge == null ? "Edge --" : `Edge ${formatPercent(edge, 1, true)}`;
    const quality = qualityValue(prediction);
    const warningClass = quality === "SUSPICIOUS" ? " suspicious" : "";
    const nrfi = prediction.nrfi_prob == null
      ? "NO DATA"
      : formatPercent(prediction.nrfi_prob);
    const marketHome = prediction.market_no_vig_home_prob == null
      ? "--"
      : formatPercent(prediction.market_no_vig_home_prob);
    const modelHome = prediction.premarket_model_home_prob == null
      ? "--"
      : formatPercent(prediction.premarket_model_home_prob);

    return `
      <article class="game-card${warningClass}">
        <div class="signal ${signalClass(prediction)}">
          <span>${escapeText(signalLabel(prediction))}</span>
          <span class="signal-edge">${escapeText(edgeText)}</span>
        </div>
        <div class="game-body">
          <div class="matchup-row">
            <div>
              <div class="matchup">${escapeText(prediction.away_team || "--")} @ ${escapeText(prediction.home_team || "--")}</div>
              <div class="game-time">${displayTime(prediction)}</div>
            </div>
            <div class="probability">
              ${homeProb == null ? "--" : formatPercent(homeProb)}
              <small>${escapeText(prediction.home_team || "Home")} win</small>
            </div>
          </div>

          <div class="tags">${tagHtml(prediction)}</div>

          <p class="source">
            Sources: ${escapeText(prediction.odds_source || "No verified source")}<br>
            Model home: ${modelHome} | Market home: ${marketHome}
          </p>

          <div class="market-grid">
            <div class="market">
              <span class="market-label">Odds</span>
              <span class="market-value">${displayOdds(prediction)}</span>
            </div>
            <div class="market">
              <span class="market-label">Moneyline</span>
              <span class="market-value">${displayMoneyline(prediction)}</span>
            </div>
            <div class="market">
              <span class="market-label">Spread</span>
              <span class="market-value">${displaySpread(prediction)}</span>
            </div>
            <div class="market">
              <span class="market-label">Total</span>
              <span class="market-value">${displayTotal(prediction)}</span>
            </div>
            <div class="market">
              <span class="market-label">NRFI</span>
              <span class="market-value">${nrfi}</span>
            </div>
            <div class="market">
              <span class="market-label">Integrity</span>
              <span class="market-value">${escapeText(quality)}</span>
            </div>
          </div>

          <div class="factors">
            Key factors: ${keyFactors(prediction) || "No additional factors available."}
          </div>
        </div>
      </article>
    `;
  }).join("");
}

async function loadDashboard() {
  try {
    const [predictionResponse, performanceResponse] = await Promise.all([
      fetch("/api/predictions"),
      fetch("/api/performance")
    ]);

    if (!predictionResponse.ok) {
      throw new Error(`Predictions API returned ${predictionResponse.status}`);
    }

    const predictions = await predictionResponse.json();
    renderGames(predictions);

    const generatedAt = parseUtcTimestamp(predictions.generated_at);
    const updated = generatedAt
      ? generatedAt.toLocaleString("zh-TW", {
          timeZone: "Asia/Taipei",
          year: "numeric",
          month: "numeric",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false
        })
      : "--";

    document.getElementById("update-time").textContent =
      `Updated in Taipei: ${updated}`;

    if (performanceResponse.ok) {
      renderPerformance(await performanceResponse.json());
    }
  } catch (error) {
    document.getElementById("messages").innerHTML =
      `<div class="message bad">Dashboard load failed: ${escapeText(error.message)}</div>`;
  }
}

loadDashboard();
</script>
</body>
</html>
"""


@app.api_route("/", methods=["GET", "HEAD"])
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


_SCHEDULE_TIME_CACHE: dict[str, tuple[datetime, dict[str, str]]] = {}
_SCHEDULE_TIME_CACHE_TTL = timedelta(minutes=30)


def _fetch_schedule_start_times(date_str: str) -> dict[str, str]:
    """Retrieve MLB game start times for display only, with a short cache."""
    now = datetime.now(timezone.utc)
    cached = _SCHEDULE_TIME_CACHE.get(date_str)
    if cached is not None and now - cached[0] < _SCHEDULE_TIME_CACHE_TTL:
        return cached[1]

    values: dict[str, str] = {}

    try:
        response = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": date_str},
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()

        for date_payload in payload.get("dates", []):
            for game in date_payload.get("games", []):
                game_id = game.get("gamePk")
                start_time = game.get("gameDate")
                if game_id is not None and start_time:
                    values[str(game_id)] = str(start_time)

    except Exception as exc:
        print(f"Start time enrichment failed for {date_str}: {exc}")

    _SCHEDULE_TIME_CACHE[date_str] = (now, values)
    return values


def _enrich_start_times(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach start_time for display when a prediction report omitted it."""
    rows = payload.get("today_predictions", [])
    if not isinstance(rows, list) or not rows:
        return payload

    lookup: dict[str, str] = {}
    dates = {
        str(row.get("game_date", ""))[:10]
        for row in rows
        if isinstance(row, dict) and row.get("game_date")
    }

    for date_str in sorted(dates):
        lookup.update(_fetch_schedule_start_times(date_str))

    for row in rows:
        if not isinstance(row, dict) or row.get("start_time"):
            continue

        start_time = lookup.get(str(row.get("game_id")))
        if start_time:
            row["start_time"] = start_time

    return payload


@app.get("/api/predictions")
def get_predictions():
    try:
        with REPORT_PATH.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)

        if isinstance(payload, dict):
            return _enrich_start_times(payload)

    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Static prediction report read failed: {exc}")

    if generate_predictions is None:
        return JSONResponse(
            {"error": "Prediction module not loaded"},
            status_code=503,
        )

    try:
        payload = generate_predictions()
        return _enrich_start_times(payload)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Real-time generation failed: {exc}"},
            status_code=500,
        )


def _recommendation_side(
    recommendation: str,
    home_team: str,
    away_team: str,
) -> str | None:
    """Return the recommended Moneyline side from a saved recommendation."""
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


def _bool_series(series: pd.Series) -> pd.Series:
    """Convert bool-like CSV values into booleans."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def _load_closing_moneyline() -> pd.DataFrame:
    """Load valid closing Moneyline rows from canonical market odds history."""
    required_columns = {
        "pipeline_version",
        "game_id",
        "market",
        "side",
        "odds",
        "is_closing_snapshot",
    }

    if not MARKET_ODDS_PATH.exists():
        return pd.DataFrame()

    try:
        market = pd.read_csv(MARKET_ODDS_PATH)
    except Exception as exc:
        print(f"Market odds history read failed: {exc}")
        return pd.DataFrame()

    if not required_columns.issubset(set(market.columns)):
        return pd.DataFrame()

    market = market.copy()
    market["game_id"] = market["game_id"].astype(str)
    market["odds"] = pd.to_numeric(market["odds"], errors="coerce")
    market["is_closing_snapshot"] = _bool_series(
        market["is_closing_snapshot"]
    )

    market = market[
        (market["pipeline_version"].astype(str) == CLEAN_PIPELINE_VERSION)
        & (market["market"].astype(str).str.lower() == "moneyline")
        & (market["side"].astype(str).str.lower().isin(["home", "away"]))
        & market["is_closing_snapshot"]
        & (market["odds"] > 1.0)
    ].copy()

    return market


def _moneyline_clv_metrics(clean_snapshot_rows: pd.DataFrame) -> dict[str, Any]:
    """Calculate entry-versus-closing Moneyline CLV for clean paper bets."""
    result: dict[str, Any] = {
        "avg_clv": None,
        "positive_clv_rate": None,
        "clv_samples": 0,
        "clv_message": "Waiting for closing lines",
    }

    if clean_snapshot_rows.empty:
        result["clv_message"] = "No clean snapshots available"
        return result

    closing_rows = _load_closing_moneyline()
    if closing_rows.empty:
        return result

    candidate_rows = clean_snapshot_rows[
        (
            clean_snapshot_rows["recommendation_status"]
            .astype(str)
            .str.upper()
            == "PAPER_BET"
        )
        & (
            clean_snapshot_rows["odds_quality_status"]
            .astype(str)
            .str.upper()
            == "OK"
        )
    ].copy()

    if candidate_rows.empty:
        result["clv_message"] = "No paper bet entries available"
        return result

    clv_values: list[float] = []

    for _, row in candidate_rows.iterrows():
        side = _recommendation_side(
            row.get("moneyline_recommendation", ""),
            row.get("home_team", ""),
            row.get("away_team", ""),
        )
        if side is None:
            continue

        entry_value = (
            row.get("home_moneyline_odds")
            if side == "home"
            else row.get("away_moneyline_odds")
        )
        entry_odds = pd.to_numeric(entry_value, errors="coerce")
        if pd.isna(entry_odds) or float(entry_odds) <= 1.0:
            continue

        game_closing = closing_rows[
            (closing_rows["game_id"] == str(row.get("game_id")))
            & (closing_rows["side"].astype(str).str.lower() == side)
        ]["odds"].dropna()

        if game_closing.empty:
            continue

        closing_odds = float(game_closing.median())
        entry_probability = 1.0 / float(entry_odds)
        closing_probability = 1.0 / closing_odds
        clv_values.append(closing_probability - entry_probability)

    if not clv_values:
        return result

    result["avg_clv"] = float(sum(clv_values) / len(clv_values))
    result["positive_clv_rate"] = float(
        sum(value > 0 for value in clv_values) / len(clv_values)
    )
    result["clv_samples"] = int(len(clv_values))
    result["clv_message"] = "Entry price compared with closing market median"

    return result


@app.get("/api/performance")
def get_performance() -> dict[str, Any]:
    """Return clean forward-tested performance and Moneyline CLV metrics."""
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
        "message": "No settled baseline_v2_clean samples yet",
    }

    if not SNAPSHOT_PATH.exists():
        return result

    try:
        frame = pd.read_csv(SNAPSHOT_PATH)

        required_columns = {
            "pipeline_version",
            "snapshot_valid",
            "game_id",
            "home_win",
            "displayed_home_win_pct",
            "recommendation_status",
            "odds_quality_status",
            "moneyline_recommendation",
            "home_team",
            "away_team",
            "home_moneyline_odds",
            "away_moneyline_odds",
        }

        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            result["message"] = (
                "Clean snapshot file is missing columns: "
                + ", ".join(missing_columns)
            )
            return result

        clean = frame[
            (frame["pipeline_version"].astype(str) == CLEAN_PIPELINE_VERSION)
            & (
                frame["snapshot_valid"]
                .astype(str)
                .str.lower()
                == "true"
            )
        ].copy()

        clv_result = _moneyline_clv_metrics(clean)
        result.update(clv_result)

        clean["home_win"] = pd.to_numeric(
            clean["home_win"],
            errors="coerce",
        )

        settled = clean[
            clean["home_win"].isin([0, 1])
        ].copy()

        if settled.empty:
            result["message"] = (
                "No settled baseline_v2_clean samples yet; CLV may appear after market close"
            )
            return result

        settled["home_win"] = settled["home_win"].astype(int)
        result["clean_sample_count"] = int(len(settled))
        result["total"] = int(len(settled))

        settled["displayed_home_win_pct"] = pd.to_numeric(
            settled["displayed_home_win_pct"],
            errors="coerce",
        )

        scored = settled[
            ["home_win", "displayed_home_win_pct"]
        ].dropna()

        if not scored.empty:
            result["brier"] = float(
                brier_score_loss(
                    scored["home_win"],
                    scored["displayed_home_win_pct"],
                )
            )

        paper_bets = settled[
            (
                settled["recommendation_status"]
                .astype(str)
                .str.upper()
                == "PAPER_BET"
            )
            & (
                settled["odds_quality_status"]
                .astype(str)
                .str.upper()
                == "OK"
            )
        ].copy()

        wins: list[int] = []
        profits: list[float] = []

        for _, row in paper_bets.iterrows():
            side = _recommendation_side(
                row.get("moneyline_recommendation", ""),
                row.get("home_team", ""),
                row.get("away_team", ""),
            )
            if side is None:
                continue

            odds_value = (
                row.get("home_moneyline_odds")
                if side == "home"
                else row.get("away_moneyline_odds")
            )
            odds = pd.to_numeric(odds_value, errors="coerce")
            if pd.isna(odds) or float(odds) <= 1.0:
                continue

            won = (
                int(row["home_win"] == 1)
                if side == "home"
                else int(row["home_win"] == 0)
            )
            wins.append(won)
            profits.append((float(odds) - 1.0) if won else -1.0)

        if profits:
            result["moneyline_bets"] = int(len(profits))
            result["win_rate"] = float(sum(wins) / len(wins))
            result["roi"] = float(sum(profits) / len(profits))

        result["message"] = (
            "Statistics for settled baseline_v2_clean snapshots"
        )

    except Exception as exc:
        result["message"] = f"Performance calculation error: {exc}"
        print(result["message"])

    return result


@app.post("/run")
def run_background(authorization: str = Header(None)) -> dict[str, str]:
    if not authorization or authorization != f"Bearer {ADMIN_TOKEN}":
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
    if not authorization or authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    return {
        "status": "disabled",
        "message": "NRFI training is currently disabled",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
