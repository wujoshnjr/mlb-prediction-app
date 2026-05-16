import sys, os, json, traceback, threading
from datetime import datetime
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from model import UnifiedSportsModel

app = FastAPI(title="MLB Prediction Engine")
model = UnifiedSportsModel()

# ---------- 前端 HTML ----------
FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB 預測分析</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #1e3c72; }
        .card { background: white; border-radius: 12px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #1e3c72; color: white; }
        .value-bet { background: #d4edda; font-weight: bold; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 14px; }
        .bet-home { background: #007bff; color: white; }
        .bet-away { background: #6f42c1; color: white; }
        .loading { text-align: center; color: #666; }
        .error { color: #dc3545; }
    </style>
</head>
<body>
    <h1>⚾ MLB 預測分析中心</h1>
    <div id="last-updated">載入中...</div>
    <div class="card">
        <h2>📊 球隊戰力排名</h2>
        <table id="rankings-table">
            <thead><tr><th>排名</th><th>球隊</th><th>勝-負</th><th>勝率</th></tr></thead>
            <tbody id="rankings-body"></tbody>
        </table>
    </div>
    <div class="card">
        <h2>📅 今日對戰預測</h2>
        <table id="predictions-table">
            <thead><tr><th>主隊</th><th>客隊</th><th>預測主勝%</th><th>預測客勝%</th><th>推薦</th></tr></thead>
            <tbody id="predictions-body"></tbody>
        </table>
    </div>
    <div class="card">
        <h2>💎 價值投注推薦</h2>
        <div id="recommendations"></div>
    </div>

    <script>
        async function loadData() {
            try {
                const resp = await fetch('/api/predictions');
                const data = await resp.json();
                document.getElementById('last-updated').innerText = '更新時間：' + data.generated_at;

                // 排名
                const rankingsBody = document.getElementById('rankings-body');
                rankingsBody.innerHTML = '';
                data.power_rankings.forEach((team, idx) => {
                    rankingsBody.innerHTML += `<tr>
                        <td>${idx+1}</td>
                        <td>${team.name}</td>
                        <td>${team.wins}-${team.losses}</td>
                        <td>${(team.win_pct*100).toFixed(1)}%</td>
                    </tr>`;
                });

                // 預測
                const predBody = document.getElementById('predictions-body');
                predBody.innerHTML = '';
                data.today_predictions.forEach(p => {
                    predBody.innerHTML += `<tr>
                        <td>${p.home_team}</td>
                        <td>${p.away_team}</td>
                        <td>${(p.predicted_home_win_pct*100).toFixed(1)}%</td>
                        <td>${(p.predicted_away_win_pct*100).toFixed(1)}%</td>
                        <td>${p.recommendation || '-'}</td>
                    </tr>`;
                });

                // 推薦
                const recDiv = document.getElementById('recommendations');
                const bets = data.today_predictions.filter(p => p.recommendation);
                if (bets.length === 0) {
                    recDiv.innerHTML = '<p>今日暫無明顯價值投注</p>';
                } else {
                    recDiv.innerHTML = bets.map(b => `<p>${b.recommendation}</p>`).join('');
                }
            } catch (err) {
                document.getElementById('last-updated').innerHTML = '<span class="error">無法載入數據，請稍後再試</span>';
            }
        }
        loadData();
    </script>
</body>
</html>
"""

# ---------- 路由 ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return FRONTEND_HTML

@app.get("/api/predictions")
def api_predictions():
    """讀取最新預測 JSON 並回傳"""
    try:
        with open('report/prediction.json', 'r') as f:
            data = json.load(f)
        return data
    except:
        # 嘗試即時生成（輕量版）
        try:
            from prediction import generate_predictions
            generate_predictions()
            with open('report/prediction.json', 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/run")
def run_background():
    """背景抓取並生成預測"""
    def task():
        from prediction import generate_predictions
        generate_predictions()
    thread = threading.Thread(target=task)
    thread.start()
    return {"status": "started", "message": "預測生成中，請稍後刷新頁面"}

# 健康檢查
@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}
