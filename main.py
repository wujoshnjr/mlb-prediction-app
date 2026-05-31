# main.py
"""Premium FastAPI dashboard for MLB full-game market analytics.

This version keeps the existing prediction report contract and adds:
- Premium dark mobile-first dashboard layout.
- Explicit Taipei-time rendering for report timestamps and game starts.
- Full-game Moneyline, Spread and Total presentation only.
- Market integrity and bookmaker source visibility.
- Clean forward-snapshot performance metrics.
- Moneyline CLV metrics derived from entry prices versus closing prices.
- Data health API and dashboard cards for pipeline visibility.

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
DAILY_CONTEXT_PATH = Path("data/daily_game_context.csv")
TRAINING_STATUS_PATH = Path("data/training_status.json")
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

.health-grid {
  display:grid;
  grid-template-columns:repeat(6, minmax(0, 1fr));
  gap:10px;
}

.health-card {
  min-height:96px;
  padding:14px;
  border-radius:17px;
  background:linear-gradient(150deg, rgba(20,27,56,.96), rgba(13,17,38,.98));
  border:1px solid var(--border);
}

.health-card.ok {
  border-color:rgba(53,239,157,.23);
  background:linear-gradient(145deg, rgba(53,239,157,.11), rgba(13,17,38,.96));
}

.health-card.warn {
  border-color:rgba(255,197,94,.28);
  background:linear-gradient(145deg, rgba(255,197,94,.11), rgba(13,17,38,.96));
}

.health-card.bad {
  border-color:rgba(255,101,119,.30);
  background:linear-gradient(145deg, rgba(255,101,119,.11), rgba(13,17,38,.96));
}

.health-label {
  color:var(--muted);
  font-size:.63rem;
  font-weight:700;
  letter-spacing:.12em;
  text-transform:uppercase;
}

.health-value {
  margin-top:9px;
  font-size:1.34rem;
  line-height:1;
  font-weight:760;
  letter-spacing:-.035em;
}

.health-caption {
  margin-top:8px;
  color:var(--muted-2);
  font-size:.66rem;
  line-height:1.35;
}

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
  .health-grid { grid-template-columns:repeat(3,minmax(0,1fr)); }
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

  .stats,
  .health-grid {
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:8px;
  }

  .stat {
    min-height:91px;
    padding:12px 11px 10px;
    border-radius:14px;
  }

  .health-card {
    min-height:88px;
    padding:12px 11px 10px;
    border-radius:14px;
  }

  .stat-value,
  .health-value {
    font-size:1.24rem;
  }

  .stat-caption,
  .health-caption {
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

  <div class="section-heading">
    <h2>Accuracy diagnostics</h2>
    <span>Breakdown of settled clean predictions</span>
  </div>

  <section class="stats">
    <div class="stat">
      <div class="stat-label">All Settled</div>
      <div id="acc-all" class="stat-value neutral">--</div>
      <div id="acc-all-caption" class="stat-caption">all clean predictions</div>
    </div>
    <div class="stat featured">
      <div class="stat-label">ML Bets</div>
      <div id="acc-ml-bets" class="stat-value neutral">--</div>
      <div id="acc-ml-bets-caption" class="stat-caption">paper bet sample</div>
    </div>
    <div class="stat">
      <div class="stat-label">Home Picks</div>
      <div id="acc-home" class="stat-value neutral">--</div>
      <div id="acc-home-caption" class="stat-caption">model home picks</div>
    </div>
    <div class="stat">
      <div class="stat-label">Away Picks</div>
      <div id="acc-away" class="stat-value neutral">--</div>
      <div id="acc-away-caption" class="stat-caption">model away picks</div>
    </div>
    <div class="stat">
      <div class="stat-label">Favorites</div>
      <div id="acc-favorites" class="stat-value neutral">--</div>
      <div id="acc-favorites-caption" class="stat-caption">market favorite picks</div>
    </div>
    <div class="stat">
      <div class="stat-label">Underdogs</div>
      <div id="acc-underdogs" class="stat-value neutral">--</div>
      <div id="acc-underdogs-caption" class="stat-caption">market underdog picks</div>
    </div>
  </section>
  
  <div class="section-heading">
    <h2>Data health</h2>
    <span>Pipeline reliability and source coverage</span>
  </div>

  <section id="health-grid" class="health-grid">
    <div class="health-card">
      <div class="health-label">Status</div>
      <div id="health-status" class="health-value waiting">Loading</div>
      <div id="health-status-caption" class="health-caption">checking pipeline</div>
    </div>
    <div class="health-card">
      <div class="health-label">Predictions</div>
      <div id="health-predictions" class="health-value neutral">--</div>
      <div id="health-predictions-caption" class="health-caption">today board</div>
    </div>
    <div class="health-card">
      <div class="health-label">Odds</div>
      <div id="health-odds" class="health-value neutral">--</div>
      <div class="health-caption">OK / suspicious / unavailable</div>
    </div>
    <div class="health-card">
      <div class="health-label">Context Ready</div>
      <div id="health-context" class="health-value neutral">--</div>
      <div id="health-context-caption" class="health-caption">pregame context</div>
    </div>
    <div class="health-card">
      <div class="health-label">Snapshots</div>
      <div id="health-snapshots" class="health-value neutral">--</div>
      <div id="health-snapshots-caption" class="health-caption">clean / settled</div>
    </div>
    <div class="health-card">
      <div class="health-label">Market Rows</div>
      <div id="health-market" class="health-value neutral">--</div>
      <div id="health-market-caption" class="health-caption">closing ML rows</div>
    </div>
    <div class="health-card">
      <div class="health-label">Game Feed</div>
      <div id="health-game-feed" class="health-value neutral">--</div>
      <div id="health-game-feed-caption" class="health-caption">MLB feed rows</div>
    </div>
    <div class="health-card">
      <div class="health-label">Lineup Feed</div>
      <div id="health-lineup-feed" class="health-value neutral">--</div>
      <div id="health-lineup-feed-caption" class="health-caption">batting order coverage</div>
    </div>
    <div class="health-card">
      <div class="health-label">Weather / Umpire</div>
      <div id="health-weather-umpire" class="health-value neutral">--</div>
      <div id="health-weather-umpire-caption" class="health-caption">weather / plate umpire</div>
    </div>
    <div class="health-card">
      <div class="health-label">Model Readiness</div>
      <div id="health-training" class="health-value neutral">--</div>
      <div id="health-training-caption" class="health-caption">training status</div>
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

function setHealthCard(id, mode) {
  const card = document.getElementById(id)?.closest(".health-card");
  if (!card) return;
  card.classList.remove("ok", "warn", "bad");
  if (mode) card.classList.add(mode);
}

function renderHealth(healthData) {
  const status = String(healthData.status || "ERROR").toUpperCase();
  const statusEl = document.getElementById("health-status");
  const captionEl = document.getElementById("health-status-caption");

  statusEl.textContent = status;
  statusEl.className = `health-value ${
    status === "OK" ? "positive" : status === "WARNING" ? "waiting" : "negative"
  }`;
  setHealthCard("health-status", status === "OK" ? "ok" : status === "WARNING" ? "warn" : "bad");

  const messages = Array.isArray(healthData.messages) ? healthData.messages : [];
  captionEl.textContent = messages.length ? messages.slice(0, 2).join(" | ") : "pipeline checks passed";

  document.getElementById("health-predictions").textContent =
    healthData.prediction_count ?? "--";
  document.getElementById("health-predictions-caption").textContent =
    `${healthData.scheduled_game_count ?? 0} scheduled games`;

  const odds = healthData.odds || {};
  document.getElementById("health-odds").textContent =
    `${odds.ok ?? 0} / ${odds.suspicious ?? 0} / ${odds.unavailable ?? 0}`;
  setHealthCard(
    "health-odds",
    (odds.suspicious || odds.unavailable || odds.missing) ? "warn" : "ok"
  );

  const context = healthData.daily_context || {};
  document.getElementById("health-context").textContent =
    `${context.ready_context_count ?? 0}`;
  document.getElementById("health-context-caption").textContent =
    `${context.latest_context_count ?? 0} latest context rows`;
  setHealthCard("health-context", context.file_exists ? "ok" : "warn");

  const latestContextCount = Number(context.latest_context_count || 0);
  const gameFeedCount = Number(context.game_feed_available_count || 0);
  const lineupPlayerCountRows = Number(context.lineup_player_count_rows || 0);
  const top3AvailableCount = Number(context.top3_available_count || 0);
  const weatherAvailableCount = Number(context.weather_available_count || 0);
  const umpireAvailableCount = Number(context.umpire_available_count || 0);

  document.getElementById("health-game-feed").textContent =
    `${gameFeedCount} / ${latestContextCount}`;
  document.getElementById("health-game-feed-caption").textContent =
    "MLB feed available";
  setHealthCard(
    "health-game-feed",
    latestContextCount > 0 && gameFeedCount >= latestContextCount ? "ok" : "warn"
  );

  document.getElementById("health-lineup-feed").textContent =
    `${lineupPlayerCountRows} / ${latestContextCount}`;
  document.getElementById("health-lineup-feed-caption").textContent =
    `${top3AvailableCount} with top 3 IDs`;
  setHealthCard(
    "health-lineup-feed",
    lineupPlayerCountRows > 0 ? "ok" : "warn"
  );

  document.getElementById("health-weather-umpire").textContent =
    `${weatherAvailableCount} / ${umpireAvailableCount}`;
  document.getElementById("health-weather-umpire-caption").textContent =
    "weather / plate umpire";
  setHealthCard(
    "health-weather-umpire",
    weatherAvailableCount > 0 || umpireAvailableCount > 0 ? "ok" : "warn"
  );
  
  const snapshots = healthData.snapshots || {};
  document.getElementById("health-snapshots").textContent =
    `${snapshots.clean_rows ?? 0} / ${snapshots.settled_rows ?? 0}`;
  document.getElementById("health-snapshots-caption").textContent =
    `${snapshots.stored_rows ?? 0} stored rows`;
  setHealthCard("health-snapshots", snapshots.file_exists ? "ok" : "warn");

  const market = healthData.market_odds_history || {};
  document.getElementById("health-market").textContent =
    `${market.stored_rows ?? 0}`;
  document.getElementById("health-market-caption").textContent =
    `${market.closing_moneyline_rows ?? 0} closing ML rows`;
  setHealthCard("health-market", market.file_exists ? "ok" : "warn");

  const training = healthData.training || {};
  const sampleCount = Number(training.sample_count || 0);
  const minimumRequired = Number(training.minimum_required || 0);
  const remaining = training.remaining_samples;

  document.getElementById("health-training").textContent =
    minimumRequired > 0 ? `${sampleCount} / ${minimumRequired}` : "--";

  if (training.trained) {
    document.getElementById("health-training-caption").textContent =
      `${training.model_type || "model"} ready`;
    setHealthCard("health-training", "ok");
  } else {
    const remainingText = remaining == null
      ? "training not ready"
      : `${remaining} more needed`;
    document.getElementById("health-training-caption").textContent =
      `Manual baseline active - ${remainingText}`;
    setHealthCard("health-training", "warn");
  }
}

function accuracyText(bucket) {
  if (!bucket || bucket.accuracy == null) return "--";
  return formatPercent(bucket.accuracy, 1);
}

function accuracyCaption(bucket, fallback) {
  if (!bucket) return fallback;
  const sampleCount = bucket.sample_count ?? 0;
  const correct = bucket.correct ?? 0;
  return `${correct} / ${sampleCount} correct`;
}

function setAccuracyValue(id, bucket) {
  const element = document.getElementById(id);
  if (!element) return;

  element.textContent = accuracyText(bucket);
  const value = bucket && bucket.accuracy != null ? Number(bucket.accuracy) : null;

  if (value == null) {
    element.className = "stat-value waiting";
  } else if (value >= 0.55) {
    element.className = "stat-value positive";
  } else if (value < 0.50) {
    element.className = "stat-value negative";
  } else {
    element.className = "stat-value neutral";
  }
}

function renderAccuracyDiagnostics(performanceData) {
  const breakdown = performanceData.accuracy_breakdown || {};

  setAccuracyValue("acc-all", breakdown.all_settled);
  document.getElementById("acc-all-caption").textContent =
    accuracyCaption(breakdown.all_settled, "all clean predictions");

  setAccuracyValue("acc-ml-bets", breakdown.moneyline_paper_bets);
  document.getElementById("acc-ml-bets-caption").textContent =
    accuracyCaption(breakdown.moneyline_paper_bets, "paper bet sample");

  setAccuracyValue("acc-home", breakdown.home_model_picks);
  document.getElementById("acc-home-caption").textContent =
    accuracyCaption(breakdown.home_model_picks, "model home picks");

  setAccuracyValue("acc-away", breakdown.away_model_picks);
  document.getElementById("acc-away-caption").textContent =
    accuracyCaption(breakdown.away_model_picks, "model away picks");

  setAccuracyValue("acc-favorites", breakdown.favorites);
  document.getElementById("acc-favorites-caption").textContent =
    accuracyCaption(breakdown.favorites, "market favorite picks");

  setAccuracyValue("acc-underdogs", breakdown.underdogs);
  document.getElementById("acc-underdogs-caption").textContent =
    accuracyCaption(breakdown.underdogs, "market underdog picks");
}

function renderPerformance(performanceData) {
  document.getElementById("total").textContent = performanceData.total ?? "--";

  const roi = document.getElementById("roi");
  if (performanceData.roi == null) {
    roi.textContent = "No bets";
    roi.className = "stat-value waiting";
  } else {
    roi.textContent = formatPercent(performanceData.roi, 1, true);
    roi.className = `stat-value ${performanceData.roi >= 0 ? "positive" : "negative"}`;
  }

  document.getElementById("roi-caption").textContent =
    `${performanceData.moneyline_bets || 0} settled ML bets`;

  const winRate = document.getElementById("win-rate");
  if (performanceData.win_rate == null) {
    winRate.textContent = "--";
    winRate.className = "stat-value waiting";
  } else {
    winRate.textContent = formatPercent(performanceData.win_rate);
    winRate.className = "stat-value neutral";
  }

  const brier = document.getElementById("brier");
  if (performanceData.brier == null) {
    brier.textContent = "--";
    brier.className = "stat-value waiting";
  } else {
    brier.textContent = Number(performanceData.brier).toFixed(3);
    brier.className = "stat-value neutral";
  }

  const avgClv = document.getElementById("avg-clv");
  const positiveClv = document.getElementById("positive-clv");

  if (performanceData.avg_clv == null) {
    avgClv.textContent = "Waiting";
    avgClv.className = "stat-value waiting";
    positiveClv.textContent = "--";
    positiveClv.className = "stat-value waiting";
    document.getElementById("clv-caption").textContent =
      performanceData.clv_message || "closing lines pending";
  } else {
    avgClv.textContent = formatPercent(performanceData.avg_clv, 2, true);
    avgClv.className = `stat-value ${performanceData.avg_clv >= 0 ? "positive" : "negative"}`;
    positiveClv.textContent = formatPercent(performanceData.positive_clv_rate);
    positiveClv.className = `stat-value ${performanceData.positive_clv_rate >= 0.5 ? "positive" : "negative"}`;
    document.getElementById("clv-caption").textContent =
      `${performanceData.clv_samples || 0} entry vs close samples`;
  }

  document.getElementById("positive-clv-caption").textContent =
    `${performanceData.clv_samples || 0} CLV samples`;
}

function renderGames(predictionData) {
  const rows = predictionData.today_predictions || [];
  const target = document.getElementById("games");
  const messages = [];

  if (!rows.length) {
    target.innerHTML = '<div class="empty">No game data available today.</div>';
    return;
  }

  const suspiciousCount = rows.filter(row => qualityValue(row) === "SUSPICIOUS").length;
  const trackingCount = rows.filter(row => statusValue(row) === "TRACKING_ONLY").length;
  const errors = predictionData.errors || [];

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

    const blockReason = prediction.recommendation_block_reason || "";
    const blockDetails = Array.isArray(prediction.recommendation_block_details)
      ? prediction.recommendation_block_details
      : [];
    const blockDetailsText = blockDetails.length
      ? blockDetails.slice(0, 2).map(escapeText).join(" - ")
      : "";
      
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
            Recommendation note: ${escapeText(blockReason || "No recommendation note.")}
            ${blockDetailsText ? `<br>${blockDetailsText}` : ""}
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
  let predictionData = null;
  let performanceData = null;
  let healthData = null;
  const dashboardMessages = [];

  try {
    const predictionResponse = await fetch("/api/predictions");
    if (!predictionResponse.ok) {
      throw new Error(`Predictions API returned ${predictionResponse.status}`);
    }

    predictionData = await predictionResponse.json();
    renderGames(predictionData);

    const generatedAt = parseUtcTimestamp(predictionData.generated_at);
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
  } catch (error) {
    dashboardMessages.push(
      `<div class="message bad">Predictions load failed: ${escapeText(error.message)}</div>`
    );
  }

  try {
    const performanceResponse = await fetch("/api/performance");
    if (!performanceResponse.ok) {
      throw new Error(`Performance API returned ${performanceResponse.status}`);
    }

    performanceData = await performanceResponse.json();
    renderPerformance(performanceData);
    renderAccuracyDiagnostics(performanceData);
  } catch (error) {
    dashboardMessages.push(
      `<div class="message bad">Performance load failed: ${escapeText(error.message)}</div>`
    );
  }

  try {
    const healthResponse = await fetch("/api/health");
    if (!healthResponse.ok) {
      throw new Error(`Health API returned ${healthResponse.status}`);
    }

    healthData = await healthResponse.json();
    renderHealth(healthData);
  } catch (error) {
    renderHealth({
      status: "ERROR",
      prediction_count: 0,
      scheduled_game_count: 0,
      odds: {ok: 0, suspicious: 0, unavailable: 0, missing: 0},
   daily_context: {
     file_exists: false,
     latest_context_count: 0,
     ready_context_count: 0,
     game_feed_available_count: 0,
     starting_pitcher_id_count: 0,
     lineup_player_count_rows: 0,
     top3_available_count: 0,
     weather_available_count: 0,
     umpire_available_count: 0
   },
      snapshots: {file_exists: false, stored_rows: 0, clean_rows: 0, settled_rows: 0},
      market_odds_history: {file_exists: false, stored_rows: 0, closing_moneyline_rows: 0},
      training: {
        file_exists: false,
        trained: false,
        skipped: false,
        sample_count: 0,
        minimum_required: 0,
        remaining_samples: null,
        model_type: "",
        reason: ""
      },
      messages: ["Health API failed"]
    });

    dashboardMessages.push(
      `<div class="message bad">Health load failed: ${escapeText(error.message)}</div>`
    );
  }

  if (dashboardMessages.length) {
    document.getElementById("messages").innerHTML = dashboardMessages.join("");
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


def _sanitize_json_value(value: Any) -> Any:
    """Convert non-JSON-safe values before returning API responses."""
    if value is None:
        return None

    if isinstance(value, float):
        if pd.isna(value):
            return None
        if value == float("inf") or value == float("-inf"):
            return None
        return value

    if isinstance(value, dict):
        return {
            str(key): _sanitize_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]

    return value
    

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
        return JSONResponse(
            {"error": "Prediction module not loaded"},
            status_code=503,
        )

    try:
        payload = generate_predictions()
        payload = _enrich_start_times(payload)
        payload = _sanitize_json_value(payload)
        return JSONResponse(content=payload)
    except Exception as exc:
        print(f"Real-time generation failed: {exc}")
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


def _read_csv_safe(path: Path) -> tuple[pd.DataFrame, str | None]:
    """Read a CSV file without raising to API callers."""
    if not path.exists():
        return pd.DataFrame(), "missing"

    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def _read_prediction_report_safe() -> tuple[dict[str, Any] | None, str | None]:
    """Read the static prediction report without generating a live report."""
    if not REPORT_PATH.exists():
        return None, "missing"

    try:
        with REPORT_PATH.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except Exception as exc:
        return None, str(exc)

    if not isinstance(payload, dict):
        return None, "report is not a JSON object"

    return payload, None

def _read_json_safe(path: Path) -> tuple[dict[str, Any], str | None]:
    """Read a JSON object without raising to API callers."""
    if not path.exists():
        return {}, "missing"

    try:
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except Exception as exc:
        return {}, str(exc)

    if not isinstance(payload, dict):
        return {}, "json payload is not an object"

    return payload, None


def _status_priority(value: str) -> int:
    """Return severity rank for OK/WARNING/ERROR."""
    return {"OK": 0, "WARNING": 1, "ERROR": 2}.get(value, 0)


def _raise_health_status(current: str, candidate: str) -> str:
    """Return the more severe health status."""
    return candidate if _status_priority(candidate) > _status_priority(current) else current


def _today_from_report(report: dict[str, Any]) -> str:
    """Infer the report date using generated_at, game_date or UTC today."""
    generated_at = str(report.get("generated_at") or "").strip()
    if generated_at:
        try:
            parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d")
        except ValueError:
            pass

    rows = report.get("today_predictions", [])
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("game_date"):
                return str(row.get("game_date"))[:10]

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _count_present_rows(frame: pd.DataFrame, date_str: str | None = None) -> pd.DataFrame:
    """Return rows filtered by game_date when possible."""
    if frame.empty or date_str is None or "game_date" not in frame.columns:
        return frame

    return frame[frame["game_date"].astype(str).str[:10] == date_str].copy()


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    """Return data-health diagnostics for dashboard and monitoring."""
    result: dict[str, Any] = {
        "status": "OK",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prediction_report_exists": False,
        "prediction_count": 0,
        "schedule_fetch_ok": None,
        "scheduled_game_count": 0,
        "errors_count": 0,
        "errors_sample": [],
        "odds": {
            "ok": 0,
            "suspicious": 0,
            "unavailable": 0,
            "missing": 0,
        },
        "recommendations": {
            "paper_bet": 0,
            "tracking_only": 0,
            "no_bet": 0,
        },
        "daily_context": {
            "file_exists": False,
            "stored_rows": 0,
            "latest_context_count": 0,
            "ready_context_count": 0,
            "bullpen_available_count": 0,
            "confirmed_lineup_count": 0,
            "game_feed_available_count": 0,
            "starting_pitcher_id_count": 0,
            "lineup_player_count_rows": 0,
            "top3_available_count": 0,
            "weather_available_count": 0,
            "umpire_available_count": 0,
        },
        "snapshots": {
            "file_exists": False,
            "stored_rows": 0,
            "clean_rows": 0,
            "settled_rows": 0,
        },
        "market_odds_history": {
            "file_exists": False,
            "stored_rows": 0,
            "moneyline_rows": 0,
            "closing_moneyline_rows": 0,
        },
        "training": {
            "file_exists": False,
            "trained": False,
            "skipped": False,
            "sample_count": 0,
            "minimum_required": 0,
            "remaining_samples": None,
            "model_type": "",
            "training_source": "",
            "reason": "",
            "timestamp": "",
        },
        "messages": [],
    }

    messages: list[str] = result["messages"]

    try:
        report, report_error = _read_prediction_report_safe()
        if report is None:
            result["status"] = "ERROR"
            messages.append(f"Prediction report unavailable: {report_error}")
            return result

        result["prediction_report_exists"] = True
        report_date = _today_from_report(report)

        predictions = report.get("today_predictions", [])
        if not isinstance(predictions, list):
            predictions = []
            result["status"] = _raise_health_status(result["status"], "ERROR")
            messages.append("today_predictions is not a list")

        result["prediction_count"] = int(len(predictions))
        result["schedule_fetch_ok"] = report.get("schedule_fetch_ok")
        scheduled_count = report.get("scheduled_game_count", 0)
        try:
            result["scheduled_game_count"] = int(scheduled_count or 0)
        except (TypeError, ValueError):
            result["scheduled_game_count"] = 0
            result["status"] = _raise_health_status(result["status"], "WARNING")
            messages.append("scheduled_game_count is not numeric")

        errors = report.get("errors", [])
        if isinstance(errors, list):
            result["errors_count"] = int(len(errors))
            result["errors_sample"] = [str(item) for item in errors[:3]]
        elif errors:
            result["errors_count"] = 1
            result["errors_sample"] = [str(errors)]

        for prediction in predictions:
            if not isinstance(prediction, dict):
                result["odds"]["missing"] += 1
                result["recommendations"]["no_bet"] += 1
                continue

            quality = str(
                prediction.get("odds_quality_status") or "UNAVAILABLE"
            ).strip().upper()
            if quality == "OK":
                result["odds"]["ok"] += 1
            elif quality == "SUSPICIOUS":
                result["odds"]["suspicious"] += 1
            elif quality == "UNAVAILABLE":
                result["odds"]["unavailable"] += 1
            else:
                result["odds"]["missing"] += 1

            recommendation_status = str(
                prediction.get("recommendation_status") or "TRACKING_ONLY"
            ).strip().upper()
            moneyline_recommendation = str(
                prediction.get("moneyline_recommendation") or "NO BET"
            ).strip().upper()

            if recommendation_status == "PAPER_BET":
                result["recommendations"]["paper_bet"] += 1
            elif recommendation_status == "TRACKING_ONLY":
                result["recommendations"]["tracking_only"] += 1
            elif moneyline_recommendation in {"NO BET", "NO DATA", "PASS", ""}:
                result["recommendations"]["no_bet"] += 1
            else:
                result["recommendations"]["tracking_only"] += 1

        if result["errors_count"] > 0:
            result["status"] = _raise_health_status(result["status"], "WARNING")
            messages.append(f"Prediction report has {result['errors_count']} warning/error entries")

        if result["schedule_fetch_ok"] is not True and result["prediction_count"] > 0:
            result["status"] = _raise_health_status(result["status"], "WARNING")
            messages.append("Predictions exist but schedule_fetch_ok is not true")

        if result["scheduled_game_count"] > 0 and result["prediction_count"] == 0:
            result["status"] = _raise_health_status(result["status"], "ERROR")
            messages.append("Scheduled games exist but today_predictions is empty")

        if (
            result["odds"]["suspicious"] > 0
            or result["odds"]["unavailable"] > 0
            or result["odds"]["missing"] > 0
        ):
            result["status"] = _raise_health_status(result["status"], "WARNING")
            messages.append("Some games have suspicious, unavailable or missing odds")

        context, context_error = _read_csv_safe(DAILY_CONTEXT_PATH)
        if context_error is None:
            result["daily_context"]["file_exists"] = True
            result["daily_context"]["stored_rows"] = int(len(context))
            latest_context = _count_present_rows(context, report_date)

            if not latest_context.empty:
                if {"game_id", "captured_at"}.issubset(set(latest_context.columns)):
                    latest_context = latest_context.copy()
                    latest_context["captured_at_parsed"] = pd.to_datetime(
                        latest_context["captured_at"],
                        errors="coerce",
                        utc=True,
                    )
                    latest_context.sort_values("captured_at_parsed", inplace=True)
                    latest_context = latest_context.groupby("game_id", as_index=False).tail(1)

                result["daily_context"]["latest_context_count"] = int(len(latest_context))

                if "context_ready_for_betting" in latest_context.columns:
                    result["daily_context"]["ready_context_count"] = int(
                        _bool_series(latest_context["context_ready_for_betting"]).sum()
                    )

                if "bullpens_ready" in latest_context.columns:
                    result["daily_context"]["bullpen_available_count"] = int(
                        _bool_series(latest_context["bullpens_ready"]).sum()
                    )
                elif {
                    "home_bullpen_data_available",
                    "away_bullpen_data_available",
                }.issubset(set(latest_context.columns)):
                    bullpen_ready = (
                        _bool_series(latest_context["home_bullpen_data_available"])
                        & _bool_series(latest_context["away_bullpen_data_available"])
                    )
                    result["daily_context"]["bullpen_available_count"] = int(bullpen_ready.sum())

                if "lineups_ready" in latest_context.columns:
                    result["daily_context"]["confirmed_lineup_count"] = int(
                        _bool_series(latest_context["lineups_ready"]).sum()
                    )
                elif {
                    "home_lineup_confirmed",
                    "away_lineup_confirmed",
                }.issubset(set(latest_context.columns)):
                    lineup_ready = (
                        _bool_series(latest_context["home_lineup_confirmed"])
                        & _bool_series(latest_context["away_lineup_confirmed"])
                    )
                    result["daily_context"]["confirmed_lineup_count"] = int(lineup_ready.sum())

                if "game_feed_available" in latest_context.columns:
                    result["daily_context"]["game_feed_available_count"] = int(
                        _bool_series(latest_context["game_feed_available"]).sum()
                    )

                if {
                    "home_starting_pitcher_id",
                    "away_starting_pitcher_id",
                }.issubset(set(latest_context.columns)):
                    home_sp_ids = latest_context["home_starting_pitcher_id"].astype(str).str.strip()
                    away_sp_ids = latest_context["away_starting_pitcher_id"].astype(str).str.strip()
                    starting_pitcher_ids_ready = (
                        home_sp_ids.ne("")
                        & home_sp_ids.str.lower().ne("nan")
                        & away_sp_ids.ne("")
                        & away_sp_ids.str.lower().ne("nan")
                    )
                    result["daily_context"]["starting_pitcher_id_count"] = int(
                        starting_pitcher_ids_ready.sum()
                    )

                if {
                    "home_lineup_player_count",
                    "away_lineup_player_count",
                }.issubset(set(latest_context.columns)):
                    home_lineup_counts = pd.to_numeric(
                        latest_context["home_lineup_player_count"],
                        errors="coerce",
                    ).fillna(0)
                    away_lineup_counts = pd.to_numeric(
                        latest_context["away_lineup_player_count"],
                        errors="coerce",
                    ).fillna(0)
                    lineup_player_count_ready = (
                        (home_lineup_counts >= 9)
                        & (away_lineup_counts >= 9)
                    )
                    result["daily_context"]["lineup_player_count_rows"] = int(
                        lineup_player_count_ready.sum()
                    )

                if {
                    "home_top3_player_ids",
                    "away_top3_player_ids",
                }.issubset(set(latest_context.columns)):
                    home_top3 = latest_context["home_top3_player_ids"].astype(str).str.strip()
                    away_top3 = latest_context["away_top3_player_ids"].astype(str).str.strip()
                    top3_ready = (
                        home_top3.ne("")
                        & home_top3.str.lower().ne("nan")
                        & away_top3.ne("")
                        & away_top3.str.lower().ne("nan")
                    )
                    result["daily_context"]["top3_available_count"] = int(
                        top3_ready.sum()
                    )

                weather_ready = pd.Series(
                    [False] * len(latest_context),
                    index=latest_context.index,
                )
                if "weather_temp" in latest_context.columns:
                    weather_ready = weather_ready | pd.to_numeric(
                        latest_context["weather_temp"],
                        errors="coerce",
                    ).notna()
                if "weather_condition" in latest_context.columns:
                    weather_condition = (
                        latest_context["weather_condition"]
                        .astype(str)
                        .str.strip()
                    )
                    weather_ready = (
                        weather_ready
                        | (
                            weather_condition.ne("")
                            & weather_condition.str.lower().ne("nan")
                        )
                    )
                result["daily_context"]["weather_available_count"] = int(
                    weather_ready.sum()
                )

                if {
                    "umpire_home_plate_id",
                    "umpire_home_plate_name",
                }.intersection(set(latest_context.columns)):
                    umpire_ready = pd.Series(
                        [False] * len(latest_context),
                        index=latest_context.index,
                    )

                    if "umpire_home_plate_id" in latest_context.columns:
                        umpire_ids = (
                            latest_context["umpire_home_plate_id"]
                            .astype(str)
                            .str.strip()
                        )
                        umpire_ready = (
                            umpire_ready
                            | (
                                umpire_ids.ne("")
                                & umpire_ids.str.lower().ne("nan")
                            )
                        )

                    if "umpire_home_plate_name" in latest_context.columns:
                        umpire_names = (
                            latest_context["umpire_home_plate_name"]
                            .astype(str)
                            .str.strip()
                        )
                        umpire_ready = (
                            umpire_ready
                            | (
                                umpire_names.ne("")
                                & umpire_names.str.lower().ne("nan")
                            )
                        )

                    result["daily_context"]["umpire_available_count"] = int(
                        umpire_ready.sum()
                    )
        else:
            result["status"] = _raise_health_status(result["status"], "WARNING")
            messages.append(f"Daily context file unavailable: {context_error}")

        snapshots, snapshot_error = _read_csv_safe(SNAPSHOT_PATH)
        if snapshot_error is None:
            result["snapshots"]["file_exists"] = True
            result["snapshots"]["stored_rows"] = int(len(snapshots))

            if not snapshots.empty:
                clean_mask = pd.Series([True] * len(snapshots), index=snapshots.index)

                if "pipeline_version" in snapshots.columns:
                    clean_mask = clean_mask & (
                        snapshots["pipeline_version"].astype(str) == CLEAN_PIPELINE_VERSION
                    )

                if "snapshot_valid" in snapshots.columns:
                    clean_mask = clean_mask & _bool_series(snapshots["snapshot_valid"])

                clean_snapshots = snapshots[clean_mask].copy()
                result["snapshots"]["clean_rows"] = int(len(clean_snapshots))

                if "home_win" in clean_snapshots.columns:
                    home_win = pd.to_numeric(clean_snapshots["home_win"], errors="coerce")
                    result["snapshots"]["settled_rows"] = int(home_win.isin([0, 1]).sum())
                elif "settled_at" in clean_snapshots.columns:
                    result["snapshots"]["settled_rows"] = int(
                        clean_snapshots["settled_at"].notna()
                        & (clean_snapshots["settled_at"].astype(str).str.strip() != "")
                    )
        else:
            result["status"] = _raise_health_status(result["status"], "WARNING")
            messages.append(f"Prediction snapshots file unavailable: {snapshot_error}")

        market, market_error = _read_csv_safe(MARKET_ODDS_PATH)
        if market_error is None:
            result["market_odds_history"]["file_exists"] = True
            result["market_odds_history"]["stored_rows"] = int(len(market))

            required_market_columns = {"market", "is_closing_snapshot"}
            if not market.empty and required_market_columns.issubset(set(market.columns)):
                moneyline_mask = market["market"].astype(str).str.lower() == "moneyline"

                if "pipeline_version" in market.columns:
                    moneyline_mask = moneyline_mask & (
                        market["pipeline_version"].astype(str) == CLEAN_PIPELINE_VERSION
                    )

                moneyline = market[moneyline_mask].copy()
                result["market_odds_history"]["moneyline_rows"] = int(len(moneyline))
                result["market_odds_history"]["closing_moneyline_rows"] = int(
                    _bool_series(moneyline["is_closing_snapshot"]).sum()
                )
            elif not market.empty:
                result["status"] = _raise_health_status(result["status"], "WARNING")
                messages.append("Market odds file missing market or is_closing_snapshot columns")
        else:
            result["status"] = _raise_health_status(result["status"], "WARNING")
            messages.append(f"Market odds history file unavailable: {market_error}")

        training_status, training_error = _read_json_safe(TRAINING_STATUS_PATH)
        if training_error is None:
            result["training"]["file_exists"] = True
            result["training"]["trained"] = bool(training_status.get("trained", False))
            result["training"]["skipped"] = bool(training_status.get("skipped", False))
            result["training"]["model_type"] = str(training_status.get("model_type", ""))
            result["training"]["training_source"] = str(
                training_status.get("training_source", "")
            )
            result["training"]["reason"] = str(training_status.get("reason", ""))
            result["training"]["timestamp"] = str(training_status.get("timestamp", ""))

            try:
                sample_count = int(training_status.get("sample_count", 0) or 0)
            except (TypeError, ValueError):
                sample_count = 0

            try:
                minimum_required = int(
                    training_status.get("minimum_clean_train_samples", 0) or 0
                )
            except (TypeError, ValueError):
                minimum_required = 0

            result["training"]["sample_count"] = sample_count
            result["training"]["minimum_required"] = minimum_required

            if minimum_required > 0:
                result["training"]["remaining_samples"] = max(
                    minimum_required - sample_count,
                    0,
                )

            if not result["training"]["trained"]:
                result["status"] = _raise_health_status(result["status"], "WARNING")
                messages.append(
                    "ML model not trained yet: "
                    + (result["training"]["reason"] or "training not ready")
                )
        else:
            result["status"] = _raise_health_status(result["status"], "WARNING")
            messages.append(f"Training status file unavailable: {training_error}")

    except Exception as exc:
        result["status"] = "ERROR"
        messages.append(f"Health calculation error: {exc}")
        print(f"Health calculation error: {exc}")

    return result


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


def _safe_rate(wins: int, total: int) -> float | None:
    """Return wins / total or None when total is zero."""
    if total <= 0:
        return None
    return float(wins / total)


def _accuracy_bucket(sample_count: int, wins: int) -> dict[str, Any]:
    """Build a compact accuracy bucket."""
    return {
        "sample_count": int(sample_count),
        "correct": int(wins),
        "accuracy": _safe_rate(wins, sample_count),
    }


def _build_accuracy_breakdown(settled: pd.DataFrame) -> dict[str, Any]:
    """Return detailed settled prediction accuracy diagnostics.

    This is diagnostic only. It does not change model behavior,
    betting recommendations, ROI, CLV or snapshot storage.
    """
    empty_bucket = {
        "sample_count": 0,
        "correct": 0,
        "accuracy": None,
    }

    result: dict[str, Any] = {
        "all_settled": dict(empty_bucket),
        "moneyline_paper_bets": dict(empty_bucket),
        "home_model_picks": dict(empty_bucket),
        "away_model_picks": dict(empty_bucket),
        "favorites": dict(empty_bucket),
        "underdogs": dict(empty_bucket),
        "edge_0_to_1": dict(empty_bucket),
        "edge_1_to_3": dict(empty_bucket),
        "edge_3_to_5": dict(empty_bucket),
        "edge_5_plus": dict(empty_bucket),
        "notes": [],
    }

    if settled.empty:
        result["notes"].append("No settled clean snapshots available.")
        return result

    frame = settled.copy()
    frame["home_win"] = pd.to_numeric(frame["home_win"], errors="coerce")
    frame["displayed_home_win_pct"] = pd.to_numeric(
        frame["displayed_home_win_pct"],
        errors="coerce",
    )
    frame["market_no_vig_home_prob"] = pd.to_numeric(
        frame.get("market_no_vig_home_prob"),
        errors="coerce",
    )
    frame["model_edge_home"] = pd.to_numeric(
        frame.get("model_edge_home"),
        errors="coerce",
    )

    scored = frame.dropna(subset=["home_win", "displayed_home_win_pct"]).copy()
    if scored.empty:
        result["notes"].append("No settled rows with model probabilities.")
        return result

    scored["model_pick_side"] = scored["displayed_home_win_pct"].apply(
        lambda value: "home" if float(value) >= 0.5 else "away"
    )
    scored["model_pick_correct"] = (
        ((scored["model_pick_side"] == "home") & (scored["home_win"] == 1))
        | ((scored["model_pick_side"] == "away") & (scored["home_win"] == 0))
    )

    result["all_settled"] = _accuracy_bucket(
        int(len(scored)),
        int(scored["model_pick_correct"].sum()),
    )

    home_picks = scored[scored["model_pick_side"] == "home"]
    result["home_model_picks"] = _accuracy_bucket(
        int(len(home_picks)),
        int(home_picks["model_pick_correct"].sum()),
    )

    away_picks = scored[scored["model_pick_side"] == "away"]
    result["away_model_picks"] = _accuracy_bucket(
        int(len(away_picks)),
        int(away_picks["model_pick_correct"].sum()),
    )

    market_scored = scored.dropna(subset=["market_no_vig_home_prob"]).copy()
    if not market_scored.empty:
        market_scored["favorite_side"] = market_scored["market_no_vig_home_prob"].apply(
            lambda value: "home" if float(value) >= 0.5 else "away"
        )
        market_scored["model_pick_is_favorite"] = (
            market_scored["model_pick_side"] == market_scored["favorite_side"]
        )

        favorite_picks = market_scored[market_scored["model_pick_is_favorite"]]
        result["favorites"] = _accuracy_bucket(
            int(len(favorite_picks)),
            int(favorite_picks["model_pick_correct"].sum()),
        )

        underdog_picks = market_scored[~market_scored["model_pick_is_favorite"]]
        result["underdogs"] = _accuracy_bucket(
            int(len(underdog_picks)),
            int(underdog_picks["model_pick_correct"].sum()),
        )
    else:
        result["notes"].append("No market probabilities available for favorite split.")

    edge_scored = scored.dropna(subset=["model_edge_home"]).copy()
    if not edge_scored.empty:
        edge_scored["abs_edge"] = edge_scored["model_edge_home"].abs()

        edge_buckets = [
            ("edge_0_to_1", 0.00, 0.01),
            ("edge_1_to_3", 0.01, 0.03),
            ("edge_3_to_5", 0.03, 0.05),
            ("edge_5_plus", 0.05, 999.0),
        ]

        for key, lower, upper in edge_buckets:
            if key == "edge_5_plus":
                bucket = edge_scored[edge_scored["abs_edge"] >= lower]
            else:
                bucket = edge_scored[
                    (edge_scored["abs_edge"] >= lower)
                    & (edge_scored["abs_edge"] < upper)
                ]

            result[key] = _accuracy_bucket(
                int(len(bucket)),
                int(bucket["model_pick_correct"].sum()),
            )
    else:
        result["notes"].append("No model edge values available for edge buckets.")

    paper_bets = scored[
        (
            scored["recommendation_status"]
            .astype(str)
            .str.upper()
            == "PAPER_BET"
        )
        & (
            scored["odds_quality_status"]
            .astype(str)
            .str.upper()
            == "OK"
        )
    ].copy()

    paper_correct = 0
    paper_count = 0

    for _, row in paper_bets.iterrows():
        side = _recommendation_side(
            row.get("moneyline_recommendation", ""),
            row.get("home_team", ""),
            row.get("away_team", ""),
        )
        if side is None:
            continue

        paper_count += 1
        if side == "home" and int(row["home_win"]) == 1:
            paper_correct += 1
        elif side == "away" and int(row["home_win"]) == 0:
            paper_correct += 1

    result["moneyline_paper_bets"] = _accuracy_bucket(
        paper_count,
        paper_correct,
    )

    return result
    

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
        "accuracy_breakdown": {},
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
        result["accuracy_breakdown"] = _build_accuracy_breakdown(settled)

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
