"""
Web 服务入口，可通过 /run 端点触发数据抓取并返回结果
"""
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
    # 返回简短摘要，避免响应过大
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
