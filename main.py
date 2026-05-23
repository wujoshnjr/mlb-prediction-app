# main.py
import sys, os, json, threading
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from prediction import generate_predictions
    from scripts.elo import MLBElosystem
    elo_system = MLBElosystem()
except Exception as e:
    print(f"Warning: {e}")
    generate_predictions = None
    elo_system = None

app = FastAPI(title="MLB Prediction Hub")

# ==================== 前端 HTML（新增 4 欄） ====================
HTML = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB Prediction Hub</title>
    <style>
        :root { --bg:#f9fafb; --card:#fff; --text:#1a202c; --muted:#718096; --border:#e2e8f0; --accent:#2b6cb0; --positive:#38a169; --negative:#e53e3e; }
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:var(--text); padding:20px; }
        .container { max-width:1600px; margin:0 auto; }
        .header { border-bottom:1px solid var(--border); padding-bottom:16px; margin-bottom:24px; }
        .header h1 { font-size:1.8rem; font-weight:600; }
        .header p { color:var(--muted); margin-top:4px; }
        .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:24px; }
        .stat-card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:18px; }
        .stat-card .label { font-size:0.8rem; color:var(--muted); text-transform:uppercase; }
        .stat-card .value { font-size:1.8rem; font-weight:600; margin-top:4px; }
        .positive { color:var(--positive); } .negative { color:var(--negative); }
        .table-container { background:var(--card); border:1px solid var(--border); border-radius:8px; overflow-x:auto; margin-bottom:24px; }
        table { width:100%; border-collapse:collapse; font-size:0.85rem; }
        th,td { padding:10px 12px; text-align:left; border-bottom:1px solid var(--border); }
        th { font-weight:600; color:var(--muted); font-size:0.75rem; text-transform:uppercase; background:#f8fafc; }
        tr:last-child td { border-bottom:none; }
        tr:hover { background:#f0f4ff; }
        .rec { display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-weight:600; }
        .rec-bet { background:#c6f6d5; color:#22543d; }
        .rec-pass { background:#edf2f7; color:#718096; }
        .loading { text-align:center; padding:32px; color:var(--muted); }
        .footer { margin-top:32px; text-align:center; color:var(--muted); font-size:0.8rem; }
        .error { color:var(--negative); background:#fff5f5; border:1px solid var(--negative); padding:12px; border-radius:8px; margin-bottom:16px; }
        /* 新增欄位樣式 */
        .nrfi-high { color: var(--positive); font-weight:600; }
        .nrfi-low { color: var(--negative); font-weight:600; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚾ MLB Prediction Hub</h1>
            <p>Probabilistic forecasts · Backtested · Transparent</p>
            <p id="update-time" style="font-size:0.8rem; margin-top:8px;">Loading...</p>
        </div>
        <div id="error-box"></div>

        <!-- 绩效卡片 -->
        <div class="grid" id="perf-cards">
            <div class="stat-card"><div class="label">Total Predictions</div><div class="value" id="perf-total">—</div></div>
            <div class="stat-card"><div class="label">ROI (1/4 Kelly)</div><div class="value" id="perf-roi">—</div></div>
            <div class="stat-card"><div class="label">Win Rate</div><div class="value" id="perf-winrate">—</div></div>
            <div class="stat-card"><div class="label">Brier Score</div><div class="value" id="perf-brier">—</div></div>
        </div>

        <!-- 预测表格（新增 4 欄） -->
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Time</th><th>Home</th><th>Away</th><th>Pred (H)</th><th>Odds</th>
                        <th>Moneyline</th><th>Spread</th><th>Total</th>
                        <th>NRFI</th>               <!-- 新增 -->
                        <th>Glicko2 RD</th>         <!-- 新增 -->
                        <th>Matchup (H/A)</th>      <!-- 新增 -->
                        <th>Odds Trend</th>         <!-- 新增 -->
                        <th>Key Factors</th>
                    </tr>
                </thead>
                <tbody id="pred-body"><tr><td colspan="13" class="loading">Loading...</td></tr></tbody>
            </table>
        </div>

        <div class="footer">Data updated hourly · Past performance does not guarantee future results</div>
    </div>

    <script>
        async function load() {
            const errorBox = document.getElementById('error-box');
            try {
                const [predResp, perfResp] = await Promise.all([
                    fetch('/api/predictions'),
                    fetch('/api/performance')
                ]);
                if (!predResp.ok) throw new Error('Predictions API ' + predResp.status);
                const predData = await predResp.json();
                renderPredictions(predData);

                if (perfResp.ok) {
                    const perfData = await perfResp.json();
                    renderPerformance(perfData);
                }
                document.getElementById('update-time').innerText = 'Updated: ' + (predData.generated_at ? new Date(predData.generated_at).toLocaleString() : '—');
            } catch (err) {
                errorBox.innerHTML = `<div class="error">Failed to load: ${err.message}</div>`;
            }
        }

        function renderPerformance(data) {
            document.getElementById('perf-total').innerText = data.total || '—';
            const roiEl = document.getElementById('perf-roi');
            const roi = data.roi;
            if (roi != null) {
                roiEl.innerHTML = roi > 0 ? `<span class="positive">+${(roi*100).toFixed(1)}%</span>` : `<span class="negative">${(roi*100).toFixed(1)}%</span>`;
            } else {
                roiEl.innerText = '—';
            }
            document.getElementById('perf-winrate').innerText = data.win_rate != null ? (data.win_rate*100).toFixed(1)+'%' : '—';
            document.getElementById('perf-brier').innerText = data.brier != null ? data.brier.toFixed(3) : '—';
        }

        function renderPredictions(data) {
            const preds = data.today_predictions || [];
            const tbody = document.getElementById('pred-body');
            if (preds.length === 0) {
                tbody.innerHTML = '<tr><td colspan="13">今日暂无比赛</td></tr>';
                return;
            }
            tbody.innerHTML = preds.map(p => {
                const time = p.game_date ? new Date(p.game_date).toLocaleTimeString('zh-TW', {hour:'2-digit',minute:'2-digit'}) : '—';
                const pred = p.predicted_home_win_pct != null ? (p.predicted_home_win_pct*100).toFixed(1)+'%' : '—';
                const odds = p.home_odds ? p.home_odds.toFixed(2) : '—';
                const ml = p.moneyline_recommendation && p.moneyline_recommendation !== 'PASS' ? `<span class="rec rec-bet">${p.moneyline_recommendation}</span>` : `<span class="rec rec-pass">PASS</span>`;
                const spread = p.spread_recommendation && p.spread_recommendation !== 'PASS' ? `<span class="rec rec-bet">${p.spread_recommendation}</span>` : `<span class="rec rec-pass">PASS</span>`;
                const total = p.total_recommendation && p.total_recommendation !== 'PASS' ? `<span class="rec rec-bet">${p.total_recommendation}</span>` : `<span class="rec rec-pass">PASS</span>`;

                // 新增欄位：NRFI 概率
                const nrfiProb = p.nrfi_prob != null ? (p.nrfi_prob*100).toFixed(1) + '%' : '—';
                const nrfiClass = p.nrfi_prob > 0.55 ? 'nrfi-high' : (p.nrfi_prob < 0.45 ? 'nrfi-low' : '');

                // Glicko2 RD Sum
                const glickoRd = p.glicko_rd_sum != null ? p.glicko_rd_sum.toFixed(0) : '—';

                // Matchup 優勢
                const matchupHome = p.home_matchup_adv != null ? p.home_matchup_adv.toFixed(3) : '—';
                const matchupAway = p.away_matchup_adv != null ? p.away_matchup_adv.toFixed(3) : '—';
                const matchup = `${matchupHome} / ${matchupAway}`;

                // 盤口趨勢
                const oddsTrend = p.home_odds_trend != null ? p.home_odds_trend.toFixed(3) : '—';

                const factors = (p.top_features || []).join(' · ');
                return `<tr>
                    <td>${time}</td>
                    <td><strong>${p.home_team}</strong></td>
                    <td>${p.away_team}</td>
                    <td>${pred}</td>
                    <td>${odds}</td>
                    <td>${ml}</td>
                    <td>${spread}</td>
                    <td>${total}</td>
                    <td class="${nrfiClass}">${nrfiProb}</td>
                    <td>${glickoRd}</td>
                    <td>${matchup}</td>
                    <td>${oddsTrend}</td>
                    <td style="font-size:0.75rem;color:var(--muted)">${factors}</td>
                </tr>`;
            }).join('');
        }

        load();
    </script>
</body>
</html>
"""

# ==================== API 端点（全部保留） ====================

@app.api_route("/", methods=["GET", "HEAD"])
def index():
    return HTMLResponse(HTML)

@app.get("/api/predictions")
def get_predictions():
    try:
        with open("report/prediction.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        pass
    except Exception:
        pass

    if generate_predictions:
        try:
            data = generate_predictions(elo_system) if elo_system else generate_predictions()
            return data
        except Exception as e:
            return JSONResponse({"error": f"Real-time generation failed: {str(e)}"}, status_code=500)
    else:
        return JSONResponse({"error": "Prediction module not loaded"}, status_code=503)

@app.get("/api/performance")
def get_performance():
    """从 historical_predictions.csv 计算真实绩效指标"""
    history_file = "data/historical_predictions.csv"
    result = {"total": 0, "roi": 0.0, "win_rate": 0.0, "brier": 0.0}

    if not os.path.exists(history_file):
        return result

    try:
        df = pd.read_csv(history_file)
        df = df[df['home_win'].notna()]
        df['home_win'] = df['home_win'].astype(int)
        if len(df) == 0:
            return result

        result["total"] = len(df)

        bets = df[df['ml_rec'].notna() & (df['ml_rec'] != '')]
        bets_with_rec = bets[bets['ml_rec'].str.contains('Bet', na=False)]
        if len(bets_with_rec) > 0:
            result["win_rate"] = bets_with_rec['home_win'].mean()

        def calc_profit(row):
            if 'Bet' not in str(row.get('ml_rec', '')):
                return 0
            odds = row.get('home_odds', 2.0)
            if pd.isna(odds) or odds <= 1:
                odds = 2.0
            return (odds - 1) if row['home_win'] == 1 else -1

        df['profit'] = df.apply(calc_profit, axis=1)
        total_bets = (df['profit'] != 0).sum()
        total_profit = df['profit'].sum()
        if total_bets > 0:
            result["roi"] = total_profit / total_bets

        clean = df[['home_win', 'pred_home_win']].dropna()
        if len(clean) > 0:
            result["brier"] = brier_score_loss(clean['home_win'], clean['pred_home_win'])

    except Exception as e:
        print(f"Performance calculation error: {e}")

    return result

@app.get("/run")
def run_background():
    def task():
        try:
            if generate_predictions:
                generate_predictions(elo_system) if elo_system else generate_predictions()
        except Exception as e:
            print(f"Background error: {e}")
    threading.Thread(target=task).start()
    return {"status": "started"}

@app.get("/health")
def health():
    return {"status": "ok"}
