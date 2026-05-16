import sys
import os
import json
import traceback
import threading
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# 尝试导入预测模块和ELO系统
try:
    from prediction import generate_predictions
    from scripts.elo import MLBElosystem
    elo_system = MLBElosystem()          # 全局ELO实例，保持状态
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
    <title>MLB 预测分析中心</title>
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
        .recommendation { background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
        .no-rec { color: #64748b; }
        .elo-badge { background: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 12px; font-size: 0.85rem; }
        .value-positive { color: #16a34a; font-weight: bold; }
        .loading { text-align: center; padding: 40px; color: #64748b; }
        .error { color: #dc2626; background: #fee2e2; padding: 15px; border-radius: 8px; }
        .flex { display: flex; gap: 20px; flex-wrap: wrap; }
        .flex > div { flex: 1; min-width: 280px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚾ MLB 预测分析中心</h1>
            <p id="update-time">载入中...</p>
        </div>

        <div id="error-box"></div>

        <div class="flex">
            <div class="card">
                <h2>📊 球队战力排名</h2>
                <table id="rankings-table">
                    <thead><tr><th>#</th><th>球队</th><th>胜-负</th><th>胜率</th><th>ELO</th></tr></thead>
                    <tbody id="rankings-body"><tr><td colspan="5" class="loading">⏳ 加载中...</td></tr></tbody>
                </table>
            </div>
            <div class="card">
                <h2>📈 ELO 评分榜</h2>
                <table id="elo-table">
                    <thead><tr><th>球队</th><th>ELO 评分</th></tr></thead>
                    <tbody id="elo-body"><tr><td colspan="2" class="loading">⏳ 加载中...</td></tr></tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <h2>📅 今日对战预测</h2>
            <table id="predictions-table">
                <thead><tr><th>主队</th><th>客队</th><th>预测主胜</th><th>预测客胜</th><th>主队ELO</th><th>客队ELO</th><th>推荐</th></tr></thead>
                <tbody id="predictions-body"><tr><td colspan="7" class="loading">⏳ 加载中...</td></tr></tbody>
            </table>
        </div>

        <div class="card">
            <h2>💎 价值投注推荐</h2>
            <div id="recommendations-box"></div>
        </div>
    </div>

    <script>
        async function loadData() {
            const errorBox = document.getElementById('error-box');
            try {
                const resp = await fetch('/api/predictions');
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const data = await resp.json();

                document.getElementById('update-time').innerText = '🕒 更新时间：' + data.generated_at;

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
                    rankingsBody.innerHTML = '<tr><td colspan="5">暂无球队数据</td></tr>';
                }

                // ELO 表格
                const eloBody = document.getElementById('elo-body');
                if (data.elo_ratings) {
                    const sorted = Object.entries(data.elo_ratings).sort((a,b) => b[1] - a[1]);
                    eloBody.innerHTML = sorted.map(([name, elo]) => `
                        <tr><td>${name}</td><td><span class="elo-badge">${elo.toFixed(1)}</span></td></tr>
                    `).join('');
                } else {
                    eloBody.innerHTML = '<tr><td colspan="2">暂无 ELO 数据</td></tr>';
                }

                // 预测表格
                const predBody = document.getElementById('predictions-body');
                if (data.today_predictions && data.today_predictions.length > 0) {
                    predBody.innerHTML = data.today_predictions.map(p => `
                        <tr>
                            <td>${p.home_team}</td>
                            <td>${p.away_team}</td>
                            <td>${(p.predicted_home_win_pct*100).toFixed(1)}%</td>
                            <td>${(p.predicted_away_win_pct*100).toFixed(1)}%</td>
                            <td><span class="elo-badge">${p.elo_home?.toFixed(0) ?? '—'}</span></td>
                            <td><span class="elo-badge">${p.elo_away?.toFixed(0) ?? '—'}</span></td>
                            <td>${p.recommendation ? `<span class="recommendation">${p.recommendation}</span>` : '<span class="no-rec">—</span>'}</td>
                        </tr>
                    `).join('');
                } else {
                    predBody.innerHTML = '<tr><td colspan="7">今日暂无比赛或数据</td></tr>';
                }

                // 推荐汇总
                const recBox = document.getElementById('recommendations-box');
                const recs = data.today_predictions?.filter(p => p.recommendation) || [];
                if (recs.length === 0) {
                    recBox.innerHTML = '<p class="no-rec">📭 今日暂无明显价值投注机会</p>';
                } else {
                    recBox.innerHTML = '<ul style="list-style:none; padding:0;">' + recs.map(p => `
                        <li style="margin-bottom:10px; padding:10px; background:#f0fdf4; border-radius:8px;">
                            <strong>${p.home_team} vs ${p.away_team}</strong><br>
                            <span class="value-positive">${p.recommendation}</span>
                        </li>
                    `).join('') + '</ul>';
                }

            } catch (err) {
                errorBox.innerHTML = `<div class="error">⚠️ 数据加载失败：${err.message}。请稍后刷新重试。</div>`;
            }
        }
        loadData();
    </script>
</body>
</html>
"""

# ==================== API 端点 ====================

@app.get("/", response_class=HTMLResponse)
def index():
    """返回前端页面"""
    return HTML

@app.get("/api/predictions")
def get_predictions():
    """
    优先读取 report/prediction.json（由 GitHub Actions 定时生成）。
    若文件不存在，则尝试实时生成（可能较慢且消耗内存，请确保 Render 资源足够）。
    """
    # 1. 尝试从文件读取
    try:
        with open("report/prediction.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        pass
    except Exception as e:
        # 文件损坏等其他错误
        pass

    # 2. 文件不存在时实时生成（备用方案）
    if generate_predictions is not None:
        try:
            data = generate_predictions(elo_system) if elo_system else generate_predictions()
            return data
        except Exception as e:
            return JSONResponse(
                {"error": f"实时生成预测失败: {str(e)}", "traceback": traceback.format_exc().split("\n")},
                status_code=500
            )
    else:
        return JSONResponse(
            {"error": "预测模块未加载，且无本地数据。请先运行 GitHub Actions 生成 prediction.json"},
            status_code=503
        )

@app.get("/run")
def run_background():
    """触发后台生成预测（不等待完成）"""
    def task():
        try:
            if generate_predictions:
                generate_predictions(elo_system) if elo_system else generate_predictions()
        except Exception as e:
            print(f"Background prediction error: {e}")
    thread = threading.Thread(target=task)
    thread.start()
    return {"status": "started", "message": "预测生成已在后台启动，请稍后刷新页面查看结果。"}

@app.get("/health")
def health():
    return {"status": "ok"}
