import sys
import os
import traceback
import requests

# === 终极全局请求头伪装 ===
def initialize_requests():
    # 设置默认 User-Agent
    requests.utils.default_user_agent = lambda: (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # 保存原始 Session 类
    OriginalSession = requests.Session

    # 自定义 Session，自动添加伪装头
    class CustomSession(OriginalSession):
        def __init__(self):
            super().__init__()
            self.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.fangraphs.com/',
            })

    # 替换全局 Session 类
    requests.Session = CustomSession
    # 同时替换 requests.get / post 使用的默认 session
    requests.get = CustomSession().get
    requests.post = CustomSession().post

initialize_requests()
# === 伪装结束 ===

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
