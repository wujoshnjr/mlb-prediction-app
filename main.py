import sys, os, json, threading
from datetime import datetime
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from scripts.database import init_database, get_performance_metrics, get_connection

try:
    from prediction import generate_predictions
    from scripts.elo import MLBElosystem
    elo_system = MLBElosystem()
except Exception as e:
    print(f"Warning: Could not import prediction/elo: {e}")
    generate_predictions = None
    elo_system = None

app = FastAPI(title="MLB Prediction Engine")

init_database()

# ========== 专业风格前端 ==========
HTML = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB Prediction Hub</title>
    <style>
        :root {
            --bg: #f9fafb;
            --card-bg: #ffffff;
            --text: #1a202c;
            --muted: #718096;
            --border: #e2e8f0;
            --accent: #2b6cb0;
            --accent-light: #ebf4ff;
            --positive: #38a169;
            --negative: #e53e3e;
            --warning: #d69e2e;
        }
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); }
        .container { max-width: 1280px; margin: 0 auto; padding: 24px 16px; }
        .header { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
        .header h1 { font-size: 1.8rem; font-weight: 600; color: #1a202c; }
        .header p { color: var(--muted); margin-top: 4px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .stat-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
        .stat-card .label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
        .stat-card .value { font-size: 2rem; font-weight: 600; margin-top: 4px; }
        .positive { color: var(--positive); }
        .negative { color: var(--negative); }
        .table-container { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
        th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); }
        th { font-weight: 600; color: var(--muted); font-size: 0.8rem; text-transform: uppercase; }
        tr:last-child td { border-bottom: none; }
        tr:hover { background: var(--accent-light); }
        .rec { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
        .rec-bet { background: #c6f6d5; color: #22543d; }
        .rec-pass { background: #edf2f7; color: #718096; }
        .tab-bar { display: flex; gap: 8px; margin-bottom: 16px; }
        .tab { padding: 8px 16px; border: 1px solid var(--border); border-radius: 4px; cursor: pointer; font-size: 0.85rem; background: var(--card-bg); }
        .tab.active { background: var(--accent); color: white; border-color: var(--accent); }
        .loading { text-align: center; padding: 32px; color: var(--muted); }
        .footer { margin-top: 32px; text-align: center; color: var(--muted); font-size: 0.8rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MLB Prediction Hub</h1>
            <p>Probabilistic forecasts · Backtested · Transparent</p>
            <p id="update-time" style="font-size:0.8rem; margin-top:8px;">Loading...</p>
        </div>

        <!-- 绩效卡片 -->
        <div class="grid" id="perf-cards">
            <div class="stat-card"><div class="label">Total Predictions</div><div class="value" id="perf-total">—</div></div>
            <div class="stat-card"><div class="label">ROI (1/4 Kelly)</div><div class="value" id="perf-roi">—</div></div>
            <div class="stat-card"><div class="label">Win Rate</div><div class="value" id="perf-winrate">—</div></div>
            <div class="stat-card"><div class="label">Brier Score</div><div class="value" id="perf-brier">—</div></div>
        </div>

        <!-- 导航 -->
        <div class="tab-bar">
            <div class="tab active" onclick="switchTab('predictions')">Predictions</div>
            <div class="tab" onclick="switchTab('rankings')">Power Rankings</div>
        </div>

        <!-- 预测表格 -->
        <div class="table-container" id="tab-predictions">
            <table>
                <thead><tr><th>Time</th><th>Home</th><th>Away</th><th>Pred (H)</th><th>Odds</th><th>Key Factors</th><th>Rec</th></tr></thead>
                <tbody id="pred-body"><tr><td colspan="7" class="loading">Loading...</td></tr></tbody>
            </table>
        </div>

        <!-- 战力排名 -->
        <div class="table-container" id="tab-rankings" style="display:none">
            <table>
                <thead><tr><th>#</th><th>Team</th><th>W-L</th><th>Win%</th><th>ELO</th></tr></thead>
                <tbody id="rank-body"><tr><td colspan="5" class="loading">Loading...</td></tr></tbody>
            </table>
        </div>

        <div class="footer">Data updated hourly · Past performance does not guarantee future results</div>
    </div>

    <script>
        let allData = null;
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab:nth-child(${tab === 'predictions' ? 1 : 2})`).classList.add('active');
            document.getElementById('tab-predictions').style.display = tab === 'predictions' ? 'block' : 'none';
            document.getElementById('tab-rankings').style.display = tab === 'rankings' ? 'block' : 'none';
        }
        async function load() {
            try {
                const [predResp, perfResp] = await Promise.all([
                    fetch('/api/predictions'),
                    fetch('/api/performance')
                ]);
                const predData = await predResp.json();
                const perfData = await perfResp.json();
                allData = predData;
                renderPredictions(predData);
                renderPerformance(perfData);
            } catch(e) {
                document.getElementById('pred-body').innerHTML = '<tr><td colspan="7" class="loading">Failed to load data</td></tr>';
            }
        }
        function renderPerformance(data) {
            document.getElementById('perf-total').innerText = data.total || '—';
            document.getElementById('perf-roi').innerHTML = data.roi ? `<span class="${data.roi > 0 ? 'positive' : 'negative'}">${(data.roi*100).toFixed(1)}%</span>` : '—';
            document.getElementById('perf-winrate').innerText = data.win_rate ? (data.win_rate*100).toFixed(1)+'%' : '—';
            document.getElementById('perf-brier').innerText = data.brier ? data.brier.toFixed(3) : '—';
            document.getElementById('update-time').innerText = 'Updated: ' + (predData?.generated_at ? new Date(predData.generated_at).toLocaleString() : '—');
        }
        function renderPredictions(data) {
            const preds = data.today_predictions || [];
            const body = document.getElementById('pred-body');
            body.innerHTML = preds.map(p => `
                <tr>
                    <td>${new Date(p.game_date).toLocaleTimeString('zh-TW', {hour:'2-digit',minute:'2-digit'})}</td>
                    <td><strong>${p.home_team}</strong></td>
                    <td>${p.away_team}</td>
                    <td>${(p.predicted_home_win_pct*100).toFixed(1)}%</td>
                    <td>${p.home_odds ? p.home_odds.toFixed(2) : '—'}</td>
                    <td style="font-size:0.8rem;color:#718096;">${(p.top_features||[]).join(' · ')}</td>
                    <td><span class="rec ${p.moneyline_recommendation !== 'PASS' ? 'rec-bet' : 'rec-pass'}">${p.moneyline_recommendation !== 'PASS' ? 'Bet' : 'Pass'}</span></td>
                </tr>
            `).join('');
            // 排名
            const rankBody = document.getElementById('rank-body');
            const ranks = data.power_rankings || [];
            rankBody.innerHTML = ranks.map((t,i) => `
                <tr>
                    <td>${i+1}</td>
                    <td>${t.name}</td>
                    <td>${t.wins}-${t.losses}</td>
                    <td>${(t.win_pct*100).toFixed(1)}%</td>
                    <td>${data.elo_ratings?.[t.name]?.toFixed(0) || '—'}</td>
                </tr>
            `).join('');
        }
        load();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

@app.get("/api/predictions")
def get_predictions():
    try:
        with open("report/prediction.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except:
        if generate_predictions:
            data = generate_predictions(elo_system) if elo_system else generate_predictions()
            return data
        return JSONResponse({"error": "No data"}, status_code=503)

@app.get("/api/performance")
def get_performance():
    try:
        metrics = get_performance_metrics()
        # 补充 Brier 计算（可从 backtest 读取，这里简化）
        return {
            "total": metrics.get("total", 0),
            "roi": 0.0,  # 需从 backtest 计算，这里占位
            "win_rate": metrics.get("win_rate", 0) or 0,
            "brier": 0.0
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
def health():
    return {"status": "ok"}
