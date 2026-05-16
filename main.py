import sys
import os
import json
import traceback
import threading
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# 嘗試導入預測模塊和ELO系統
try:
    from prediction import generate_predictions
    from scripts.elo import MLBElosystem
    elo_system = MLBElosystem()          # 全局ELO實例，保持狀態
except Exception as e:
    print(f"Warning: Could not import prediction/elo: {e}")
    generate_predictions = None
    elo_system = None

app = FastAPI(title="MLB Prediction Engine")

# ==================== 前端 HTML ====================
HTML = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB 預測分析中心</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f4f8; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; border-radius: 16px; margin-bottom: 20px; }
        .header h1 { font-size: 2.5rem; margin-bottom: 5px; }
        .header p { opacity: 0.9; }
        .card { background: white; border-radius: 16px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
        .card h2 { color: #1e3c72; margin-bottom: 15px; border-bottom: 2px solid #e0e7ff; padding-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8fafc; color: #1e3c72; font-weight: 600; }
        tr:hover { background: #f1f5f9; }
        .recommendation { background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 20px; font-weight: 600; display: inline-block; margin: 2px 0; }
        .rec-spread { background: #dbeafe; color: #1e40af; padding: 4px 12px; border-radius: 20px; font-weight: 600; display: inline-block; margin: 2px 0; }
        .rec-total { background: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 20px; font-weight: 600; display: inline-block; margin: 2px 0; }
        .no-rec { color: #64748b; }
        .elo-badge { background: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 12px; font-size: 0.85rem; }
        .value-positive { color: #16a34a; font-weight: bold; }
        .loading { text-align: center; padding: 40px; color: #64748b; }
        .error { color: #dc2626; background: #fee2e2; padding: 15px; border-radius: 8px; }
        .flex { display: flex; gap: 20px; flex-wrap: wrap; }
        .flex > div { flex: 1; min-width: 280px; }
        .nav-tabs { display: flex; gap: 10px; margin-bottom: 20px; }
        .nav-tab { padding: 10px 20px; background: white; border-radius: 8px; cursor: pointer; font-weight: 600; color: #1e3c72; border: 2px solid #e0e7ff; }
        .nav-tab.active { background: #1e3c72; color: white; border-color: #1e3c72; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .summary-box { display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px; }
        .summary-item { flex: 1; min-width: 150px; background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        .summary-item .number { font-size: 2rem; font-weight: bold; color: #1e3c72; }
        .summary-item .label { color: #64748b; font-size: 0.9rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚾ MLB 預測分析中心</h1>
            <p>🎯 全方位運彩投注決策系統 | 勝負盤 · 讓分盤 · 大小分盤 | 蒙地卡羅模擬</p>
            <p id="update-time" style="margin-top:10px;">載入中...</p>
        </div>

        <div id="error-box"></div>

        <!-- 摘要卡片 -->
        <div class="summary-box" id="summary-box"></div>

        <!-- 導航標籤 -->
        <div class="nav-tabs">
            <div class="nav-tab active" onclick="switchTab('all')">📋 全部推薦</div>
            <div class="nav-tab" onclick="switchTab('moneyline')">💰 勝負盤</div>
            <div class="nav-tab" onclick="switchTab('spread')">🎯 讓分盤</div>
            <div class="nav-tab" onclick="switchTab('total')">📏 大小分盤</div>
            <div class="nav-tab" onclick="switchTab('rankings')">📊 戰力排名</div>
        </div>

        <!-- 全部推薦 -->
        <div class="tab-content active" id="tab-all">
            <div class="card">
                <h2>📅 今日對戰預測總覽</h2>
                <table id="predictions-table">
                    <thead><tr><th>主隊</th><th>客隊</th><th>預測主勝</th><th>預測客勝</th><th>主ELO</th><th>客ELO</th><th>勝負推薦</th><th>讓分推薦</th><th>大小推薦</th></tr></thead>
                    <tbody id="predictions-body"><tr><td colspan="9" class="loading">⏳ 加載中...</td></tr></tbody>
                </table>
            </div>
        </div>

        <!-- 勝負盤 -->
        <div class="tab-content" id="tab-moneyline">
            <div class="card">
                <h2>💰 勝負盤 (Moneyline) 推薦</h2>
                <table id="moneyline-table">
                    <thead><tr><th>比賽</th><th>預測主勝</th><th>賠率</th><th>凱利值</th><th>推薦</th></tr></thead>
                    <tbody id="moneyline-body"><tr><td colspan="5" class="loading">⏳ 加載中...</td></tr></tbody>
                </table>
            </div>
        </div>

        <!-- 讓分盤 -->
        <div class="tab-content" id="tab-spread">
            <div class="card">
                <h2>🎯 讓分盤 (Spread) 推薦</h2>
                <table id="spread-table">
                    <thead><tr><th>比賽</th><th>讓分線</th><th>主隊過盤率</th><th>客隊過盤率</th><th>推薦</th></tr></thead>
                    <tbody id="spread-body"><tr><td colspan="5" class="loading">⏳ 加載中...</td></tr></tbody>
                </table>
            </div>
        </div>

        <!-- 大小分盤 -->
        <div class="tab-content" id="tab-total">
            <div class="card">
                <h2>📏 大小分盤 (Total) 推薦</h2>
                <table id="total-table">
                    <thead><tr><th>比賽</th><th>大小分線</th><th>模擬總分</th><th>大分機率</th><th>小分機率</th><th>推薦</th></tr></thead>
                    <tbody id="total-body"><tr><td colspan="6" class="loading">⏳ 加載中...</td></tr></tbody>
                </table>
            </div>
        </div>

        <!-- 戰力排名 -->
        <div class="tab-content" id="tab-rankings">
            <div class="flex">
                <div class="card">
                    <h2>📊 球隊戰力排名</h2>
                    <table id="rankings-table">
                        <thead><tr><th>#</th><th>球隊</th><th>勝-負</th><th>勝率</th><th>ELO</th></tr></thead>
                        <tbody id="rankings-body"><tr><td colspan="5" class="loading">⏳ 加載中...</td></tr></tbody>
                    </table>
                </div>
                <div class="card">
                    <h2>📈 ELO 評分榜</h2>
                    <table id="elo-table">
                        <thead><tr><th>球隊</th><th>ELO 評分</th></tr></thead>
                        <tbody id="elo-body"><tr><td colspan="2" class="loading">⏳ 加載中...</td></tr></tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- 模擬詳情 -->
        <div class="card">
            <h2>🎲 蒙地卡羅模擬詳情 (5,000次)</h2>
            <table id="sim-table">
                <thead><tr><th>比賽</th><th>模擬平均總分</th><th>模擬平均分差</th><th>80% 置信區間 (分差)</th></tr></thead>
                <tbody id="sim-body"><tr><td colspan="4" class="loading">⏳ 加載中...</td></tr></tbody>
            </table>
        </div>

        <!-- 歷史回測 -->
        <div class="card">
            <h2>📈 歷史回測績效</h2>
            <p id="backtest-status">回測數據需手動執行 backtest.py 生成</p>
            <div id="backtest-content"></div>
        </div>
    </div>

    <script>
        let allData = null;

        function switchTab(tabId) {
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`.nav-tab[onclick="switchTab('${tabId}')"]`).classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.add('active');
        }

        async function loadData() {
            const errorBox = document.getElementById('error-box');
            try {
                const resp = await fetch('/api/predictions');
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                allData = await resp.json();
                renderAll(allData);
            } catch (err) {
                errorBox.innerHTML = `<div class="error">⚠️ 數據加載失敗：${err.message}。請稍後刷新重試。</div>`;
            }
        }

        function renderAll(data) {
            document.getElementById('update-time').innerText = '🕒 更新時間：' + data.generated_at;

            // 摘要卡片
            const summaryBox = document.getElementById('summary-box');
            const mlBets = data.bet_summary?.moneyline_bets?.length || 0;
            const spreadBets = data.bet_summary?.spread_bets?.length || 0;
            const totalBets = data.bet_summary?.total_bets?.length || 0;
            const totalGames = data.today_predictions?.length || 0;
            summaryBox.innerHTML = `
                <div class="summary-item"><div class="number">${totalGames}</div><div class="label">今日比賽</div></div>
                <div class="summary-item"><div class="number">${mlBets}</div><div class="label">💰 勝負推薦</div></div>
                <div class="summary-item"><div class="number">${spreadBets}</div><div class="label">🎯 讓分推薦</div></div>
                <div class="summary-item"><div class="number">${totalBets}</div><div class="label">📏 大小推薦</div></div>
            `;

            // 總覽表格
            const predBody = document.getElementById('predictions-body');
            if (data.today_predictions && data.today_predictions.length > 0) {
                predBody.innerHTML = data.today_predictions.map(p => `
                    <tr>
                        <td><strong>${p.home_team}</strong></td>
                        <td>${p.away_team}</td>
                        <td>${(p.predicted_home_win_pct*100).toFixed(1)}%</td>
                        <td>${(p.predicted_away_win_pct*100).toFixed(1)}%</td>
                        <td><span class="elo-badge">${p.elo_home?.toFixed(0) ?? '—'}</span></td>
                        <td><span class="elo-badge">${p.elo_away?.toFixed(0) ?? '—'}</span></td>
                        <td>${p.moneyline_recommendation !== 'PASS' ? `<span class="recommendation">${p.moneyline_recommendation}</span>` : '<span class="no-rec">—</span>'}</td>
                        <td>${p.spread_recommendation !== 'PASS' ? `<span class="rec-spread">${p.spread_recommendation}</span>` : '<span class="no-rec">—</span>'}</td>
                        <td>${p.total_recommendation !== 'PASS' ? `<span class="rec-total">${p.total_recommendation}</span>` : '<span class="no-rec">—</span>'}</td>
                    </tr>
                `).join('');
            } else {
                predBody.innerHTML = '<tr><td colspan="9">今日暫無比賽或數據</td></tr>';
            }

            // 勝負盤
            const mlBody = document.getElementById('moneyline-body');
            mlBody.innerHTML = data.today_predictions?.map(p => `
                <tr>
                    <td>${p.home_team} vs ${p.away_team}</td>
                    <td>${(p.predicted_home_win_pct*100).toFixed(1)}%</td>
                    <td>${p.home_odds?.toFixed(2) ?? '—'}</td>
                    <td>${p.kelly_fraction ? (p.kelly_fraction*100).toFixed(1)+'%' : '—'}</td>
                    <td>${p.moneyline_recommendation !== 'PASS' ? `<span class="recommendation">${p.moneyline_recommendation}</span>` : '<span class="no-rec">—</span>'}</td>
                </tr>
            `).join('') || '<tr><td colspan="5">無數據</td></tr>';

            // 讓分盤
            const spreadBody = document.getElementById('spread-body');
            spreadBody.innerHTML = data.today_predictions?.map(p => `
                <tr>
                    <td>${p.home_team} vs ${p.away_team}</td>
                    <td>${p.spread_line}</td>
                    <td>${p.home_cover_prob ? (p.home_cover_prob*100).toFixed(1)+'%' : '—'}</td>
                    <td>${p.away_cover_prob ? (p.away_cover_prob*100).toFixed(1)+'%' : '—'}</td>
                    <td>${p.spread_recommendation !== 'PASS' ? `<span class="rec-spread">${p.spread_recommendation}</span>` : '<span class="no-rec">—</span>'}</td>
                </tr>
            `).join('') || '<tr><td colspan="5">無數據</td></tr>';

            // 大小分盤
            const totalBody = document.getElementById('total-body');
            totalBody.innerHTML = data.today_predictions?.map(p => `
                <tr>
                    <td>${p.home_team} vs ${p.away_team}</td>
                    <td>${p.total_line}</td>
                    <td>${p.simulated_total_mean?.toFixed(1) ?? '—'}</td>
                    <td>${p.over_prob ? (p.over_prob*100).toFixed(1)+'%' : '—'}</td>
                    <td>${p.under_prob ? (p.under_prob*100).toFixed(1)+'%' : '—'}</td>
                    <td>${p.total_recommendation !== 'PASS' ? `<span class="rec-total">${p.total_recommendation}</span>` : '<span class="no-rec">—</span>'}</td>
                </tr>
            `).join('') || '<tr><td colspan="6">無數據</td></tr>';

            // 模擬詳情
            const simBody = document.getElementById('sim-body');
            simBody.innerHTML = data.today_predictions?.map(p => `
                <tr>
                    <td>${p.home_team} vs ${p.away_team}</td>
                    <td>${p.simulated_total_mean?.toFixed(1) ?? '—'}</td>
                    <td>${p.simulated_diff_mean?.toFixed(1) ?? '—'}</td>
                    <td>[${p.confidence_interval_diff?.[0]?.toFixed(1) ?? '—'}, ${p.confidence_interval_diff?.[1]?.toFixed(1) ?? '—'}]</td>
                </tr>
            `).join('') || '<tr><td colspan="4">無模擬數據</td></tr>';

            // 排名表格
            const rankingsBody = document.getElementById('rankings-body');
            if (data.power_rankings && data.power_rankings.length > 0) {
                rankingsBody.innerHTML = data.power_rankings.map((t, i) => `
                    <tr>
                        <td>${i+1}</td>
                        <td><strong>${t.name}</strong></td>
                        <td>${t.wins}-${t.losses}</td>
                        <td>${(t.win_pct*100).toFixed(1)}%</td>
                        <td><span class="elo-badge">${data.elo_ratings?.[t.name]?.toFixed(0) ?? '—'}</span></td>
                    </tr>
                `).join('');
            } else {
                rankingsBody.innerHTML = '<tr><td colspan="5">暫無球隊數據</td></tr>';
            }

            // ELO 表格
            const eloBody = document.getElementById('elo-body');
            if (data.elo_ratings) {
                const sorted = Object.entries(data.elo_ratings).sort((a,b) => b[1] - a[1]);
                eloBody.innerHTML = sorted.map(([name, elo]) => `
                    <tr><td>${name}</td><td><span class="elo-badge">${elo.toFixed(1)}</span></td></tr>
                `).join('');
            } else {
                eloBody.innerHTML = '<tr><td colspan="2">暫無 ELO 數據</td></tr>';
            }

            // 回測 (如果有)
            fetch('/api/backtest')
                .then(r => r.json())
                .then(bt => {
                    if (bt && bt.period) {
                        document.getElementById('backtest-status').innerText = `回測期間：${bt.period}`;
                        document.getElementById('backtest-content').innerHTML = `
                            <p>💰 勝負盤：${bt.moneyline?.total_bets || 0} 筆推薦，準確率 ${bt.moneyline?.accuracy || 'N/A'}</p>
                            <p>🎯 讓分盤：${bt.spread?.total_bets || 0} 筆推薦，過盤率 ${bt.spread?.cover_rate || 'N/A'}</p>
                            <p>📏 大小分：${bt.total?.total_bets || 0} 筆推薦，準確率 ${bt.total?.accuracy || 'N/A'}</p>
                        `;
                    }
                }).catch(() => {});
        }

        loadData();
    </script>
</body>
</html>
"""

# ==================== API 端點 ====================

@app.get("/", response_class=HTMLResponse)
def index():
    """返回前端頁面"""
    return HTML

@app.get("/api/predictions")
def get_predictions():
    """
    優先讀取 report/prediction.json（由 GitHub Actions 定時生成）。
    若文件不存在，則嘗試實時生成（可能較慢且消耗內存，請確保 Render 資源足夠）。
    """
    # 1. 嘗試從文件讀取
    try:
        with open("report/prediction.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # 2. 文件不存在時實時生成（備用方案）
    if generate_predictions is not None:
        try:
            data = generate_predictions(elo_system) if elo_system else generate_predictions()
            return data
        except Exception as e:
            return JSONResponse(
                {"error": f"實時生成預測失敗: {str(e)}", "traceback": traceback.format_exc().split("\n")},
                status_code=500
            )
    else:
        return JSONResponse(
            {"error": "預測模塊未加載，且無本地數據。請先運行 GitHub Actions 生成 prediction.json"},
            status_code=503
        )

@app.get("/api/backtest")
def get_backtest():
    """讀取回測報告"""
    try:
        with open("report/backtest_report.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return JSONResponse({"message": "回測報告尚未生成，請運行 backtest.py"}, status_code=404)

@app.get("/run")
def run_background():
    """觸發後台生成預測（不等待完成）"""
    def task():
        try:
            if generate_predictions:
                generate_predictions(elo_system) if elo_system else generate_predictions()
        except Exception as e:
            print(f"Background prediction error: {e}")
    thread = threading.Thread(target=task)
    thread.start()
    return {"status": "started", "message": "預測生成已在後台啟動，請稍後刷新頁面查看結果。"}

@app.get("/health")
def health():
    return {"status": "ok"}
