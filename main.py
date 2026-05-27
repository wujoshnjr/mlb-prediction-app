# main.py
"""FastAPI dashboard for MLB predictions.

This version keeps the existing prediction report contract and adds:
- Explicit Taipei-time rendering for report timestamps and game starts.
- Market integrity status and bookmaker source display.
- Tracking-only presentation for unavailable or suspicious odds.
- Moneyline ROI scoring restricted to trusted paper-bet records when available.

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


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLB Prediction Hub</title>
<style>
:root {
  --bg:#f9fafb;
  --card:#ffffff;
  --text:#1a202c;
  --muted:#718096;
  --border:#e2e8f0;
  --positive:#38a169;
  --negative:#e53e3e;
  --warn:#b7791f;
}
* { box-sizing:border-box; }
body {
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
  background:var(--bg);
  color:var(--text);
  margin:0;
  padding:18px;
}
.container { max-width:1500px; margin:0 auto; }
.header {
  border-bottom:1px solid var(--border);
  padding-bottom:16px;
  margin-bottom:22px;
}
h1 { margin:0; font-size:2rem; }
.subtitle,.updated { color:var(--muted); margin:8px 0 0; }
.grid {
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:12px;
  margin:20px 0;
}
.card {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:10px;
  padding:18px;
}
.label { font-size:.8rem; color:var(--muted); text-transform:uppercase; }
.value { font-size:1.85rem; font-weight:650; margin-top:7px; }
.positive { color:var(--positive); }
.negative { color:var(--negative); }
.notice {
  background:#fffaf0;
  border:1px solid #f6e05e;
  border-radius:8px;
  padding:12px;
  margin:14px 0;
  color:#744210;
  font-size:.88rem;
}
.error {
  background:#fff5f5;
  border:1px solid #feb2b2;
  border-radius:8px;
  padding:12px;
  margin:14px 0;
  color:#9b2c2c;
  font-size:.88rem;
}
.table-wrap {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:10px;
  overflow-x:auto;
}
table { border-collapse:collapse; width:100%; font-size:.9rem; }
th,td {
  padding:12px;
  border-bottom:1px solid var(--border);
  text-align:left;
  white-space:nowrap;
  vertical-align:top;
}
th {
  color:var(--muted);
  font-size:.75rem;
  text-transform:uppercase;
  background:#f8fafc;
}
.rec {
  display:inline-block;
  padding:4px 9px;
  border-radius:5px;
  font-weight:650;
  font-size:.76rem;
}
.bet { background:#c6f6d5; color:#22543d; }
.pass { background:#edf2f7; color:#4a5568; }
.no-data { background:#feebc8; color:#744210; }
.status-badge {
  display:inline-block;
  padding:4px 9px;
  border-radius:999px;
  font-weight:650;
  font-size:.72rem;
  text-transform:uppercase;
}
.status-ok { background:#c6f6d5; color:#22543d; }
.status-track { background:#feebc8; color:#744210; }
.status-bad { background:#fed7d7; color:#9b2c2c; }
.integrity-detail {
  display:block;
  margin-top:5px;
  color:var(--muted);
  font-size:.7rem;
  line-height:1.35;
  white-space:normal;
  min-width:170px;
}
.integrity-warning {
  display:block;
  margin-top:5px;
  color:#9b2c2c;
  font-size:.7rem;
  line-height:1.35;
  white-space:normal;
}
.factors {
  font-size:.75rem;
  color:var(--muted);
  white-space:normal;
  min-width:180px;
}
.footer {
  margin-top:26px;
  color:var(--muted);
  font-size:.8rem;
  text-align:center;
}
.market-help {
  background:#ebf8ff;
  border:1px solid #bee3f8;
  color:#2a4365;
  border-radius:8px;
  padding:10px 12px;
  margin:12px 0 16px;
  font-size:.8rem;
  line-height:1.4;
}
.line-missing {
  color:#975a16;
  font-size:.72rem;
  display:block;
  margin-top:4px;
}
.mobile-list { display:none; }
.game-card {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:12px;
  padding:14px;
  margin-bottom:12px;
}
.game-card.suspicious { border-color:#feb2b2; }
.game-head {
  display:flex;
  justify-content:space-between;
  gap:10px;
  align-items:flex-start;
  margin-bottom:10px;
}
.matchup { font-size:1.05rem; font-weight:650; }
.game-time { color:var(--muted); font-size:.83rem; margin-top:4px; }
.probability { font-size:1.35rem; font-weight:700; white-space:nowrap; }
.probability small {
  display:block;
  color:var(--muted);
  font-size:.68rem;
  font-weight:500;
  text-align:right;
}
.mobile-status {
  display:flex;
  flex-wrap:wrap;
  align-items:center;
  gap:7px;
  margin:0 0 8px;
}
.mobile-source {
  color:var(--muted);
  font-size:.7rem;
  line-height:1.35;
  margin-bottom:10px;
}
.market-grid {
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:9px;
}
.market {
  background:#f8fafc;
  border-radius:8px;
  padding:9px;
  min-height:62px;
}
.market-label {
  display:block;
  color:var(--muted);
  font-size:.66rem;
  text-transform:uppercase;
  margin-bottom:5px;
}
.market-value { font-size:.86rem; font-weight:550; }
.mobile-factors {
  margin-top:10px;
  color:var(--muted);
  font-size:.74rem;
  line-height:1.4;
}
@media (max-width: 760px) {
  body { padding:10px; }
  h1 { font-size:1.55rem; }
  .grid {
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:8px;
    margin:14px 0;
  }
  .card { padding:12px; }
  .value { font-size:1.2rem; }
  .label { font-size:.68rem; }
  .table-wrap { display:none; }
  .mobile-list { display:block; }
}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>MLB Prediction Hub</h1>
    <p class="subtitle">Probabilistic forecasts - Backtested - Transparent</p>
    <p id="update-time" class="updated">Loading...</p>
  </div>
  <div id="messages"></div>
  <div class="market-help">
    ML = winner only. Spread and Total use the recorded line shown beside each recommendation.
    Paper Bet means odds passed integrity checks; Tracking Only means the market was unavailable or suspicious.
  </div>
  <div class="grid">
    <div class="card"><div class="label">Settled Predictions</div><div id="total" class="value">--</div></div>
    <div class="card"><div class="label">ROI (Moneyline)</div><div id="roi" class="value">--</div></div>
    <div class="card"><div class="label">Win Rate (Bets)</div><div id="win-rate" class="value">--</div></div>
    <div class="card"><div class="label">Brier Score</div><div id="brier" class="value">--</div></div>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Home</th><th>Away</th><th>Pred (H)</th><th>Odds</th>
          <th>Market Status</th><th>Moneyline</th><th>Spread</th><th>Total</th><th>NRFI</th><th>Key Factors</th>
        </tr>
      </thead>
      <tbody id="predictions"><tr><td colspan="11">Loading...</td></tr></tbody>
    </table>
  </div>
  <div id="mobile-predictions" class="mobile-list"></div>
  <div class="footer">Data updated hourly - Paper trading only - Past performance does not guarantee future results</div>
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

function parseUtcTimestamp(raw) {
  if (!raw) return null;
  const text = String(raw);
  const hasTimezone = /(?:Z|[+-]\d{2}:\d{2})$/i.test(text);
  const normalized = hasTimezone ? text : `${text}Z`;
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.valueOf()) ? null : parsed;
}

function displayTime(prediction) {
  const raw = prediction.start_time || prediction.game_datetime || prediction.game_time || prediction.game_date;
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

function displayOdds(prediction) {
  const home = prediction.home_moneyline_odds ?? prediction.home_odds;
  const away = prediction.away_moneyline_odds ?? prediction.away_odds;
  if (home == null && away == null) return "--";
  const homeText = home == null ? "--" : Number(home).toFixed(2);
  const awayText = away == null ? "--" : Number(away).toFixed(2);
  return `${escapeText(prediction.home_team || "HOME")} ${homeText} / ${escapeText(prediction.away_team || "AWAY")} ${awayText}`;
}

function signedLine(value) {
  if (value == null || value === "") return null;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return `${numeric > 0 ? "+" : ""}${numeric.toFixed(1)}`;
}

function displayMoneyline(prediction) {
  const recommendation = prediction.moneyline_recommendation;
  if (!recommendation || recommendation === "NO BET") {
    return badge(recommendation);
  }
  return `${badge(recommendation)}<span class="line-missing">Winner only</span>`;
}

function displaySpread(prediction) {
  const recommendation = prediction.spread_recommendation;
  if (!recommendation || recommendation === "NO BET" || recommendation === "NO DATA") {
    return badge(recommendation);
  }

  const homeLine = signedLine(prediction.spread_line);
  if (homeLine == null) {
    return `${badge(recommendation)}<span class="line-missing">Line not recorded</span>`;
  }

  const recommendationText = String(recommendation).toLowerCase();
  const homeTeamText = String(prediction.home_team || "").toLowerCase();
  const isHome = homeTeamText && recommendationText.includes(homeTeamText);
  const recommendedLine = isHome
    ? homeLine
    : signedLine(-Number(prediction.spread_line));
  const team = recommendation.replace(/\s+spread$/i, "");

  return badge(`${team} ${recommendedLine}`);
}

function displayTotal(prediction) {
  const recommendation = prediction.total_recommendation;
  if (!recommendation || recommendation === "NO BET" || recommendation === "NO DATA") {
    return badge(recommendation);
  }

  if (prediction.total_line == null || prediction.total_line === "") {
    return `${badge(recommendation)}<span class="line-missing">Line not recorded</span>`;
  }

  const line = Number(prediction.total_line);
  if (!Number.isFinite(line)) {
    return `${badge(recommendation)}<span class="line-missing">Line not recorded</span>`;
  }

  return badge(`${recommendation} ${line.toFixed(1)}`);
}

function qualityValue(prediction) {
  return String(prediction.odds_quality_status || "UNAVAILABLE").toUpperCase();
}

function statusValue(prediction) {
  return String(prediction.recommendation_status || "TRACKING_ONLY").toUpperCase();
}

function statusClass(prediction) {
  const quality = qualityValue(prediction);
  const status = statusValue(prediction);
  if (quality === "SUSPICIOUS") return "status-bad";
  if (quality !== "OK" || status === "TRACKING_ONLY") return "status-track";
  return "status-ok";
}

function statusLabel(prediction) {
  return statusValue(prediction) === "PAPER_BET" ? "Paper Bet" : "Tracking Only";
}

function marketStatus(prediction, includeDetails = true) {
  const quality = qualityValue(prediction);
  const sources = prediction.odds_source || "No verified source";
  const reason = prediction.suspicious_odds_reason || "";

  let html = `<span class="status-badge ${statusClass(prediction)}">${escapeText(statusLabel(prediction))}</span>`;

  if (includeDetails) {
    html += `<span class="integrity-detail">Odds: ${escapeText(quality)}<br>Sources: ${escapeText(sources)}</span>`;
    if (quality !== "OK" && reason) {
      html += `<span class="integrity-warning">${escapeText(reason)}</span>`;
    }
  }

  return html;
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

function renderPerformance(data) {
  document.getElementById("total").textContent = data.total ?? "--";

  const roiEl = document.getElementById("roi");
  if (data.roi == null) {
    roiEl.textContent = "No valid bets";
    roiEl.className = "value";
  } else {
    roiEl.textContent = `${data.roi >= 0 ? "+" : ""}${(data.roi * 100).toFixed(1)}%`;
    roiEl.className = `value ${data.roi >= 0 ? "positive" : "negative"}`;
  }

  document.getElementById("win-rate").textContent =
    data.win_rate == null ? "No valid bets" : `${(data.win_rate * 100).toFixed(1)}%`;

  document.getElementById("brier").textContent =
    data.brier == null ? "No settled samples" : Number(data.brier).toFixed(3);
}

function renderMobileCards(rows) {
  const target = document.getElementById("mobile-predictions");

  if (!rows.length) {
    target.innerHTML = '<div class="game-card">No game data available today.</div>';
    return;
  }

  target.innerHTML = rows.map(prediction => {
    const homeWin = prediction.predicted_home_win_pct == null
      ? "--"
      : `${(Number(prediction.predicted_home_win_pct) * 100).toFixed(1)}%`;

    const nrfi = prediction.nrfi_prob == null
      ? "NO DATA"
      : `${(Number(prediction.nrfi_prob) * 100).toFixed(1)}%`;

    const quality = qualityValue(prediction);
    const warningClass = quality === "SUSPICIOUS" ? " suspicious" : "";
    const reason = prediction.suspicious_odds_reason || "";

    return `<div class="game-card${warningClass}">
      <div class="game-head">
        <div>
          <div class="matchup">${escapeText(prediction.away_team || "--")} @ ${escapeText(prediction.home_team || "--")}</div>
          <div class="game-time">${displayTime(prediction)}</div>
        </div>
        <div class="probability">${homeWin}<small>${escapeText(prediction.home_team || "HOME")} WIN</small></div>
      </div>
      <div class="mobile-status">
        ${marketStatus(prediction, false)}
        <span class="status-badge ${statusClass(prediction)}">Odds ${escapeText(quality)}</span>
      </div>
      <div class="mobile-source">
        Sources: ${escapeText(prediction.odds_source || "No verified source")}
        ${quality !== "OK" && reason ? `<span class="integrity-warning">${escapeText(reason)}</span>` : ""}
      </div>
      <div class="market-grid">
        <div class="market">
          <span class="market-label">Odds</span>
          <span class="market-value">${displayOdds(prediction)}</span>
        </div>
        <div class="market">
          <span class="market-label">Moneyline</span>
          ${displayMoneyline(prediction)}
        </div>
        <div class="market">
          <span class="market-label">Spread</span>
          ${displaySpread(prediction)}
        </div>
        <div class="market">
          <span class="market-label">Total</span>
          ${displayTotal(prediction)}
        </div>
        <div class="market">
          <span class="market-label">NRFI</span>
          <span class="market-value">${nrfi}</span>
        </div>
      </div>
      <div class="mobile-factors">${keyFactors(prediction) || "No additional factors available."}</div>
    </div>`;
  }).join("");
}

function renderPredictions(data) {
  const rows = data.today_predictions || [];
  const tbody = document.getElementById("predictions");

  renderMobileCards(rows);

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="11">No game data available today.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(prediction => {
    const homeWin = prediction.predicted_home_win_pct == null
      ? "--"
      : `${(Number(prediction.predicted_home_win_pct) * 100).toFixed(1)}%`;

    const nrfi = prediction.nrfi_prob == null
      ? "--"
      : `${(Number(prediction.nrfi_prob) * 100).toFixed(1)}%`;

    return `<tr>
      <td>${displayTime(prediction)}</td>
      <td><strong>${escapeText(prediction.home_team || "--")}</strong></td>
      <td>${escapeText(prediction.away_team || "--")}</td>
      <td>${homeWin}</td>
      <td>${displayOdds(prediction)}</td>
      <td>${marketStatus(prediction)}</td>
      <td>${displayMoneyline(prediction)}</td>
      <td>${displaySpread(prediction)}</td>
      <td>${displayTotal(prediction)}</td>
      <td>${nrfi}</td>
      <td class="factors">${keyFactors(prediction)}</td>
    </tr>`;
  }).join("");

  const errors = data.errors || [];
  const trackingCount = rows.filter(row => statusValue(row) === "TRACKING_ONLY").length;
  const suspiciousCount = rows.filter(row => qualityValue(row) === "SUSPICIOUS").length;
  const messages = [];

  if (suspiciousCount) {
    messages.push(
      `<div class="error">${suspiciousCount} game(s) have suspicious odds and are tracking only. Betting recommendations are disabled for those games.</div>`
    );
  } else if (trackingCount) {
    messages.push(
      `<div class="notice">${trackingCount} game(s) are tracking only because verified odds were unavailable.</div>`
    );
  }

  if (errors.length) {
    messages.push(
      `<div class="notice">Current report contains ${errors.length} data-quality or feature warnings. Review report diagnostics before relying on recommendations.</div>`
    );
  }

  document.getElementById("messages").innerHTML = messages.join("");
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
    renderPredictions(predictions);

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

    document.getElementById("update-time").textContent = `Updated (Taipei): ${updated}`;

    if (performanceResponse.ok) {
      renderPerformance(await performanceResponse.json());
    }
  } catch (error) {
    document.getElementById("messages").innerHTML =
      `<div class="error">Dashboard load failed: ${escapeText(error.message)}</div>`;
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
def get_predictions() -> dict[str, Any] | JSONResponse:
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


def _first_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _recommendation_side(
    recommendation: str,
    home_team: str,
    away_team: str,
) -> str | None:
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


@app.get("/api/performance")
def get_performance() -> dict[str, Any]:
    result: dict[str, Any] = {
        "total": 0,
        "roi": None,
        "win_rate": None,
        "brier": None,
        "moneyline_bets": 0,
    }

    if not HISTORY_PATH.exists():
        return result

    try:
        frame = pd.read_csv(HISTORY_PATH)

        result_col = _first_column(frame, ["home_win"])
        prediction_col = _first_column(
            frame,
            ["pred_home_win", "predicted_home_win_pct"],
        )
        recommendation_col = _first_column(
            frame,
            ["ml_rec", "moneyline_recommendation"],
        )
        home_odds_col = _first_column(
            frame,
            ["home_odds", "home_moneyline_odds"],
        )
        away_odds_col = _first_column(
            frame,
            ["away_odds", "away_moneyline_odds"],
        )
        home_team_col = _first_column(frame, ["home_team"])
        away_team_col = _first_column(frame, ["away_team"])

        if result_col is None:
            return result

        frame[result_col] = pd.to_numeric(frame[result_col], errors="coerce")
        settled = frame[frame[result_col].notna()].copy()

        if settled.empty:
            return result

        settled[result_col] = settled[result_col].astype(int)
        result["total"] = int(len(settled))

        if prediction_col is not None:
            settled[prediction_col] = pd.to_numeric(
                settled[prediction_col],
                errors="coerce",
            )
            scored = settled[[result_col, prediction_col]].dropna()
            if not scored.empty:
                result["brier"] = float(
                    brier_score_loss(
                        scored[result_col],
                        scored[prediction_col],
                    )
                )

        can_score_bets = (
            recommendation_col is not None
            and home_team_col is not None
            and away_team_col is not None
            and (home_odds_col is not None or away_odds_col is not None)
        )

        if can_score_bets:
            wins: list[int] = []
            profits: list[float] = []

            for _, row in settled.iterrows():
                side = _recommendation_side(
                    row.get(recommendation_col, ""),
                    row.get(home_team_col, ""),
                    row.get(away_team_col, ""),
                )
                if side is None:
                    continue

                odds_col = home_odds_col if side == "home" else away_odds_col
                if odds_col is None:
                    continue

                odds = pd.to_numeric(row.get(odds_col), errors="coerce")
                if pd.isna(odds) or float(odds) <= 1.0:
                    continue

                won = (
                    int(row[result_col] == 1)
                    if side == "home"
                    else int(row[result_col] == 0)
                )
                wins.append(won)
                profits.append((float(odds) - 1.0) if won else -1.0)

            if profits:
                result["moneyline_bets"] = int(len(profits))
                result["win_rate"] = float(sum(wins) / len(wins))
                result["roi"] = float(sum(profits) / len(profits))

    except Exception as exc:
        print(f"Performance calculation error: {exc}")

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
