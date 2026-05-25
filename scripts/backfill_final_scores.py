# scripts/backfill_final_scores.py
import pandas as pd
import numpy as np
import requests, time, os, json
from datetime import datetime

HISTORY_FILE = "data/historical_predictions.csv"
HIST_DIR = "data/historical"

def fetch_final_score(game_id):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("gameData",{}).get("status",{}).get("abstractGameState") != "Final":
            return None, None, None
        home = data["liveData"]["linescore"]["teams"]["home"]["runs"]
        away = data["liveData"]["linescore"]["teams"]["away"]["runs"]
        return 1 if home > away else 0, home, away
    except:
        return None, None, None

def backfill():
    # 读取 CSV
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        # 找出缺失比分的行
        missing = df[df['home_score'].isna() & df['game_id'].notna()]
        print(f"CSV 缺失比分 {len(missing)} 行")
        backfilled = 0
        for idx, row in missing.iterrows():
            game_id = row['game_id']
            result = fetch_final_score(game_id)
            if result:
                home_win, home_score, away_score = result
                df.at[idx, 'home_win'] = home_win
                df.at[idx, 'home_score'] = home_score
                df.at[idx, 'away_score'] = away_score
                backfilled += 1
            time.sleep(0.5)
        df.to_csv(HISTORY_FILE, index=False)
        print(f"CSV 补回 {backfilled} 场比分")

    # 处理 parquet 文件
    if os.path.exists(HIST_DIR):
        for f in os.listdir(HIST_DIR):
            if f.endswith('.parquet'):
                path = os.path.join(HIST_DIR, f)
                pdf = pd.read_parquet(path)
                if 'home_score' in pdf.columns and pdf['home_score'].isna().any():
                    # 类似处理
                    pass

if __name__ == "__main__":
    backfill()
