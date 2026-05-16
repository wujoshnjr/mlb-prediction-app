import sys
import os
import traceback
import threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from model import UnifiedSportsModel

app = FastAPI(title="Unified Sports Data Model")
model = UnifiedSportsModel()

@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"message": "Unified Sports Model is running. Visit /run to trigger data fetch in background."}

def background_fetch():
    """在背景執行資料抓取，結果自動存入 report/ 資料夾"""
    try:
        data = model.gather_all_data()
        print("Background data fetch completed successfully.")
    except Exception as e:
        print(f"Background fetch failed: {e}")

@app.get("/run")
def run_all():
    # 立即回傳，不等待抓取完成
    thread = threading.Thread(target=background_fetch)
    thread.start()
    return {
        "status": "started",
        "message": "Data collection is running in the background. Check /report later for results."
    }

@app.get("/report")
def get_latest_report():
    """查看最新產生的報告"""
    from datetime import datetime
    date_str = datetime.now().strftime('%Y-%m-%d')
    filepath = f"report/{date_str}.json"
    if os.path.exists(filepath):
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    else:
        return {"message": f"No report found for {date_str}. It may still be generating, please try again shortly."}

# 如果需要直接測試（非背景），可保留此端點
@app.get("/run-sync")
def run_sync():
    try:
        data = model.gather_all_data()
        return {"date": data["date"], "errors": data.get("errors", [])}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc().split("\n")}
