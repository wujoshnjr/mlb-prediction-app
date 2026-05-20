import csv
import os
import requests
import time

HISTORY_FILE = "data/historical_predictions.csv"

def fetch_game_result(game_id):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        home_runs = data.get("liveData", {}).get("linescore", {}).get("teams", {}).get("home", {}).get("runs")
        away_runs = data.get("liveData", {}).get("linescore", {}).get("teams", {}).get("away", {}).get("runs")
        if home_runs is not None and away_runs is not None:
            return 1 if home_runs > away_runs else 0
    except Exception as e:
        print(f"获取比赛 {game_id} 结果失败: {e}")
    return None

def update_results():
    if not os.path.exists(HISTORY_FILE):
        print("历史预测文件不存在")
        return

    rows = []
    updated_count = 0
    with open(HISTORY_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["home_win"].strip() == "" and row["game_id"].strip():
                result = fetch_game_result(row["game_id"])
                if result is not None:
                    row["home_win"] = str(result)
                    updated_count += 1
                    print(f"✅ {row['home_team']} vs {row['away_team']}: home_win={result}")
                else:
                    print(f"⏳ {row['home_team']} vs {row['away_team']}: 比赛未结束或数据不可用")
                time.sleep(0.5)
            rows.append(row)

    with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"结果更新完成，本次更新 {updated_count} 场比赛")

if __name__ == "__main__":
    update_results()
