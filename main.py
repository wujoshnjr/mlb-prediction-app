"""
Web 服务入口，可通过 /run 端点触发数据抓取并返回结果
"""
import sys
import os

# 将当前文件所在的目录（项目根目录）添加到 Python 路径
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
    data = model.gather_all_data()
    summary = {
        "date": data["date"],
        "mlb_games": len(data["mlb_statsapi"]),
        "savant_records": len(data["savant_statcast"]),
        "retrosheet_records": len(data["retrosheet"]),
        "pybaseball_statcast_records": len(data["pybaseball_statcast"]),
        "balldontlie_teams": len(data["balldontlie_teams"]),
        "odds_records": len(data["odds_data"]),
        "weather_hours": len(data["openmeteo_weather"])
    }
    return summary
