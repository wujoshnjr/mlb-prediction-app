# scripts/backfill_final_scores.py
import pandas as pd
import numpy as np
import requests
import time
import os
import json

HISTORY_FILE = "data/historical_predictions.csv"
HIST_DIR = "data/historical"

def fetch_final_score(game_id):
    """返回 (home_win, home_score, away_score) 或 None"""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        status = data.get("gameData", {}).get("status", {})
        if status.get("abstractGameState") != "Final":
            return None
        home_runs = data.get("liveData", {}).get("linescore", {}).get("teams", {}).get("home", {}).get("runs")
        away_runs = data.get("liveData", {}).get("linescore", {}).get("teams", {}).get("away", {}).get("runs")
        if home_runs is not None and away_runs is not None:
            return 1 if home_runs > away_runs else 0, int(home_runs), int(away_runs)
    except:
        pass
    return None

def backfill():
    scanned = 0
    backfilled = 0
    skipped_non_final = 0
    api_failed = 0
    duplicates = set()

    # 处理 CSV
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        if 'home_score' not in df.columns:
            df['home_score'] = np.nan
        if 'away_score' not in df.columns:
            df['away_score'] = np.nan
        if 'home_win' not in df.columns:
            df['home_win'] = np.nan

        missing_mask = df['home_score'].isna() & df['game_id'].notna()
        missing = df[missing_mask]
        print(f"CSV missing scores: {len(missing)} rows")
        for idx, row in missing.iterrows():
            game_id = row['game_id']
            if game_id in duplicates:
                continue
            scanned += 1
            result = fetch_final_score(game_id)
            if result is not None:
                home_win, home_score, away_score = result
                df.at[idx, 'home_win'] = home_win
                df.at[idx, 'home_score'] = home_score
                df.at[idx, 'away_score'] = away_score
                backfilled += 1
                duplicates.add(game_id)
            elif result is None:
                # 可能是非 Final 或 API 失败，通过再次查询状态区分
                # 简单处理：未拿到有效结果视为 API 失败
                api_failed += 1
            time.sleep(0.3)
        df.to_csv(HISTORY_FILE, index=False)
        print(f"CSV backfilled: {backfilled}")

    # 处理 parquet 文件
    if os.path.exists(HIST_DIR):
        for fname in os.listdir(HIST_DIR):
            if fname.endswith('.parquet'):
                path = os.path.join(HIST_DIR, fname)
                pdf = pd.read_parquet(path)
                date_col = None
                if 'game_date' in pdf.columns:
                    date_col = 'game_date'
                elif 'date' in pdf.columns:
                    date_col = 'date'
                # 确保必要列存在
                if 'home_score' not in pdf.columns:
                    pdf['home_score'] = np.nan
                if 'away_score' not in pdf.columns:
                    pdf['away_score'] = np.nan
                if 'home_win' not in pdf.columns:
                    pdf['home_win'] = np.nan

                missing = pdf['home_score'].isna() & pdf['game_id'].notna()
                changed = False
                for idx, row in pdf[missing].iterrows():
                    game_id = row['game_id']
                    if game_id in duplicates:
                        continue
                    scanned += 1
                    result = fetch_final_score(game_id)
                    if result is not None:
                        home_win, home_score, away_score = result
                        pdf.at[idx, 'home_win'] = home_win
                        pdf.at[idx, 'home_score'] = home_score
                        pdf.at[idx, 'away_score'] = away_score
                        backfilled += 1
                        changed = True
                        duplicates.add(game_id)
                    else:
                        api_failed += 1
                    time.sleep(0.3)
                if changed:
                    pdf.to_parquet(path, index=False)
                    print(f"Parquet {fname} updated")

    print("===== Backfill Summary =====")
    print(f"Scanned: {scanned}")
    print(f"Backfilled: {backfilled}")
    print(f"Skipped non-final/API failed: {api_failed}")
    print(f"Duplicates ignored: {len(duplicates)}")

if __name__ == "__main__":
    backfill()
