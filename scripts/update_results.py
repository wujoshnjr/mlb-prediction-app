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
        print("历史预测文件不存在，无需更新")
        return

    rows = []
    updated_count = 0

    # 读取现有数据
    with open(HISTORY_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # 过滤掉可能存在的 None 键（如果列标题为空）
        fieldnames = [fn for fn in reader.fieldnames if fn is not None]
        for row in reader:
            # 确保 row 只包含有效字段
            clean_row = {k: v for k, v in row.items() if k in fieldnames}
            if clean_row.get("home_win", "").strip() == "" and clean_row.get("game_id", "").strip():
                result = fetch_game_result(clean_row["game_id"])
                if result is not None:
                    clean_row["home_win"] = str(result)
                    updated_count += 1
                    print(f"✅ {clean_row['home_team']} vs {clean_row['away_team']}: home_win={result}")
                else:
                    print(f"⏳ {clean_row['home_team']} vs {clean_row['away_team']}: 比赛未结束或数据不可用")
                time.sleep(0.5)
            rows.append(clean_row)

    # 写回文件（使用清理后的字段名）
    with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"结果更新完成，本次更新 {updated_count} 场比赛")

if __name__ == "__main__":
    update_results()
