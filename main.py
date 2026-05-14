"""
Web 服務入口，可透過 /run 端點觸發資料抓取並返回結果
"""
import sys
import os
import traceback

# 確保專案根目錄在 Python 路徑中
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
            "weather_hours": len(data["openmeteo_weather"])
        }
        return summary
    except Exception as e:
        # 把完整的錯誤堆疊返回，方便診斷
        error_msg = traceback.format_exc()
        return {
            "error": str(e),
            "traceback": error_msg.split("\n")
        }
