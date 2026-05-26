# main.py
import json
import os
import sys
import threading

import pandas as pd
from sklearn.metrics import brier_score_loss
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

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

HTML = r"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLB Prediction Hub</title>
<style>
:root{--bg:#f9fafb;--card:#fff;--text:#1a202c;--muted:#718096;--border:#e2e8f0;--positive:#38a169;--negative:#e53e3e;--warn:#b7791f}
*{box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:18px}
.container{max-width:1500px;margin:auto}.header{border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:22px}
h1{margin:0;font-size:2rem}.subtitle,.updated{color:var(--muted);margin:8px 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px}
.label{font-size:.8rem;color:var(--muted);text-transform:uppercase}.value{font-size:1.85rem;font-weight:650;margin-top:7px}
.positive{color:var(--positive)}.negative{color:var(--negative)}.warning{color:var(--warn)}
.notice{background:#fffaf0;border:1px solid #f6e05e;border-radius:8px;padding:12px;margin:14px 0;color:#744210;font-size:.88rem}
.error{background:#fff5f5;border:1px solid #feb2b2;border-radius:8px;padding:12px;margin:14px 0;color:#9b2c2c;font-size:.88rem}
.table-wrap{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{padding:12px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}
th{color:var(--muted);font-size:.75rem;text-transform:uppercase;background:#f8fafc}
.rec{display:inline-block;padding:4px 9px;border-radius:5px;font-weight:650;font-size:.76rem}
.bet{background:#c6f6d5;color:#22543d}.pass{background:#edf2f7;color:#4a5568}.no-data{background:#feebc8;color:#744210}
.factors{font-size:.75rem;color:var(--muted);white-space:normal;min-width:180px}
.footer{margin-top:26px;color:var(--muted);font-size:.8rem;text-align:center}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>â¾ MLB Prediction Hub</h1>
    <p class="subtitle">Probabilistic forecasts Â· Backtested Â· Transparent</p>
    <p id="update-time" class="updated">è¼å¥ä¸­...</p>
  </div>
  <div id="messages"></div>
  <div class="grid">
    <div class="card"><div class="label">Settled Predictions</div><div id="total" class="value">â</div></div>
    <div class="card"><div class="label">ROI (Moneyline)</div><div id="roi" class="value">â</div></div>
    <div class="card"><div class="label">Win Rate (Bets)</div><div id="win-rate" class="value">â</div></div>
    <div class="card"><div class="label">Brier Score</div><div id="brier" class="value">â</div></div>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Time</th><th>Home</th><th>Away</th><th>Pred (H)</th><th>Odds</th>
        <th>Moneyline</th><th>Spread</th><th>Total</th><th>NRFI</th><th>Key Factors</th>
      </tr></thead>
      <tbody id="predictions"><tr><td colspan="10">è¼å¥ä¸­...</td></tr></tbody>
    </table>
  </div>
  <div class="footer">Data updated hourly Â· Past performance does not guarantee future results</div>
</div>
<script>
function badge(value) {
  if (!value || value === "PASS" || value === "NO BET") return `<span class="rec pass">${value || "â"}</span>`;
  if (value === "NO DATA") return `<span class="rec no-data">NO DATA</span>`;
  return `<span class="rec bet">${value}</span>`;
}
function displayTime(p) {
  const raw = p.start_time || p.game_datetime || p.game_time || p.game_date;
  if (!raw) return "â";
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return "æéå¾æ´æ°";
  const parsed = new Date(raw);
  return isNaN(parsed.valueOf()) ? "â" : parsed.toLocaleTimeString("zh-TW", {hour:"2-digit", minute:"2-digit"});
}
function displayOdds(p) {
  const home = p.home_moneyline_odds ?? p.home_odds;
  const away = p.away_moneyline_odds ?? p.away_odds;
  if (home == null && away == null) return "â";
  const h = home == null ? "â" : Number(home).toFixed(2);
  const a = away == null ? "â" : Number(away).toFixed(2);
  return `H ${h} / A ${a}`;
}
function keyFactors(p) {
  if (Array.isArray(p.top_features) && p.top_features.length) return p.top_features.join(" Â· ");
  const source = p.features || {};
  return Object.entries(source)
    .filter(([_, value]) => typeof value === "number" && Math.abs(value) > 0.0001)
    .sort((a,b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0,3)
    .map(([key,value]) => `${key}=${Number(value).toFixed(2)}`)
    .join(" Â· ");
}
function renderPerformance(data) {
  document.getElementById("total").textContent = data.total ?? "â";
  const roiEl = document.getElementById("roi");
  const winEl = document.getElementById("win-rate");
  const brierEl = document.getElementById("brier");
  if (data.roi == null) roiEl.textContent = "ç¡ææææ³¨";
  else { roiEl.textContent = `${data.roi >= 0 ? "+" : ""}${(data.roi*100).toFixed(1)}%`; roiEl.className = `value ${data.roi >= 0 ? "positive" : "negative"}`; }
  winEl.textContent = data.win_rate == null ? "ç¡ææææ³¨" : `${(data.win_rate*100).toFixed(1)}%`;
  brierEl.textContent = data.brier == null ? "å°ç¡çµç®æ¨£æ¬" : Number(data.brier).toFixed(3);
}
function renderPredictions(data) {
  const rows = data.today_predictions || [];
  const tbody = document.getElementById("predictions");
  if (!rows.length) { tbody.innerHTML = `<tr><td colspan="10">ä»æ¥ç¡è³½äºè³æ</td></tr>`; return; }
  tbody.innerHTML = rows.map(p => {
    const pred = p.predicted_home_win_pct == null ? "â" : `${(Number(p.predicted_home_win_pct)*100).toFixed(1)}%`;
    const nrfi = p.nrfi_prob == null ? "â" : `${(Number(p.nrfi_prob)*100).toFixed(1)}%`;
    return `<tr>
      <td>${displayTime(p)}</td><td><strong>${p.home_team || "â"}</strong></td><td>${p.away_team || "â"}</td>
      <td>${pred}</td><td>${displayOdds(p)}</td><td>${badge(p.moneyline_recommendation)}</td>
      <td>${badge(p.spread_recommendation)}</td><td>${badge(p.total_recommendation)}</td>
      <td>${nrfi}</td><td class="factors">${keyFactors(p)}</td>
    </tr>`;
  }).join("");
  const errors = data.errors || [];
  if (errors.length) {
    document.getElementById("messages").innerHTML =
      `<div class="notice">ç®åé æ¸¬ä»æ ${errors.length} ç­è³æåè³ªï¼åè½è­¦åï¼è«ä»¥é æ¸¬æªè¨ºæ·çºæºã</div>`;
  }
}
async function loadDashboard() {
  try {
    const [predResponse, perfResponse] = await Promise.all([fetch("/api/predictions"), fetch("/api/performance")]);
    if (!predResponse.ok) throw new Error(`Predictions API ${predResponse.status}`);
    const pred = await predResponse.json();
    renderPredictions(pred);
    document.getElementById("update-time").textContent =
      `Updated: ${pred.generated_at ? new Date(pred.generated_at).toLocaleString("zh-TW") : "â"}`;
    if (perfResponse.ok) renderPerformance(await perfResponse.json());
  } catch (error) {
    document.getElementById("messages").innerHTML = `<div class="error">è¼å¥å¤±æï¼${error.message}</div>`;
  }
}
loadDashboard();
</script>
</body>
</html>
"""


@app.api_route("/", methods=["GET", "HEAD"])
def index():
    return HTMLResponse(HTML)


@app.get("/api/predictions")
def get_predictions():
    try:
        with open("report/prediction.json", "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Static prediction report read failed: {exc}")

    if generate_predictions is None:
        return JSONResponse({"error": "Prediction module not loaded"}, status_code=503)
    try:
        return generate_predictions()
    except Exception as exc:
        return JSONResponse({"error": f"Real-time generation failed: {exc}"}, status_code=500)


def _first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


@app.get("/api/performance")
def get_performance():
    result = {
        "total": 0,
        "roi": None,
        "win_rate": None,
        "brier": None,
        "moneyline_bets": 0,
    }
    history_file = "data/historical_predictions.csv"
    if not os.path.exists(history_file):
        return result

    try:
        df = pd.read_csv(history_file)
        result_col = _first_column(df, ["home_win"])
        prediction_col = _first_column(df, ["pred_home_win", "predicted_home_win_pct"])
        recommendation_col = _first_column(df, ["ml_rec", "moneyline_recommendation"])
        odds_col = _first_column(df, ["home_odds", "home_moneyline_odds"])

        if result_col is None:
            return result

        df[result_col] = pd.to_numeric(df[result_col], errors="coerce")
        settled = df[df[result_col].notna()].copy()
        if settled.empty:
            return result
        settled[result_col] = settled[result_col].astype(int)
        result["total"] = int(len(settled))

        if prediction_col is not None:
            settled[prediction_col] = pd.to_numeric(settled[prediction_col], errors="coerce")
            scored = settled[[result_col, prediction_col]].dropna()
            if not scored.empty:
                result["brier"] = float(brier_score_loss(scored[result_col], scored[prediction_col]))

        if recommendation_col is not None and odds_col is not None:
            rec_text = settled[recommendation_col].fillna("").astype(str)
            bets = settled[rec_text.str.contains("Bet", case=False, na=False)].copy()
            bets[odds_col] = pd.to_numeric(bets[odds_col], errors="coerce")
            bets = bets[bets[odds_col].notna() & (bets[odds_col] > 1.0)]
            if not bets.empty:
                def home_side(row):
                    return "home" in str(row[recommendation_col]).lower() or str(row[recommendation_col]).lower().startswith("bet ")
                wins = []
                profits = []
                for _, row in bets.iterrows():
                    is_home = home_side(row)
                    won = int(row[result_col] == 1) if is_home else int(row[result_col] == 0)
                    wins.append(won)
                    profits.append((float(row[odds_col]) - 1.0) if won else -1.0)
                result["moneyline_bets"] = int(len(profits))
                result["win_rate"] = float(sum(wins) / len(wins))
                result["roi"] = float(sum(profits) / len(profits))
    except Exception as exc:
        print(f"Performance calculation error: {exc}")

    return result


@app.post("/run")
def run_background(authorization: str = Header(None)):
    if not authorization or authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    def task():
        try:
            if generate_predictions is not None:
                generate_predictions()
        except Exception as exc:
            print(f"Background prediction error: {exc}")

    threading.Thread(target=task, daemon=True).start()
    return {"status": "started"}


@app.post("/train-nrfi")
def train_nrfi(authorization: str = Header(None)):
    if not authorization or authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": "disabled", "message": "NRFI training is currently disabled"}


@app.get("/health")
def health():
    return {"status": "ok"}
