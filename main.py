import sys
import os
import traceback
import requests

# === 全局请求头伪装（启动时立刻执行） ===
def initialize_requests():
    # 修正点：lambda 不要任何参数
    requests.utils.default_user_agent = lambda: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    # 给 requests 自身的全局 session 加上伪装头
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.fangraphs.com/',
    })
    # 让 requests.get / post 默认使用这个伪装好的 session（安全做法）
    # 注意：session.get 是绑定方法，可以直接替换顶层函数
    requests.get = session.get
    requests.post = session.post

initialize_requests()
# === 全局伪装结束 ===

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from model import UnifiedSportsModel

app = FastAPI(title="Unified Sports Data Model")
model = UnifiedSportsModel()

@app.get("/")
def read_root():
    return {"message": "Unified Sports Model is running. Visit /run to trigger data fetch."}

@app.get("/run")
def run_all():
    try:
        data = model.gather_all_data()
        summary = {
            "date": data["date"],
            "mlb_games": len(data["mlb_statsapi"]),
            "savant_records": len(data["savant_statcast"]),
            "retrosheet_records": len(data["retrosheet"]),
            "pybaseball_statcast_records": len(data["pybaseball_statcast"]),
            "balldontlie_teams": len(data["balldontlie_teams"]),
            "odds_records": len(data["odds_data"]),
            "weather_hours": len(data["openmeteo_weather"]),
            "errors": data.get("errors", [])
        }
        return summary
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc().split("\n")}
