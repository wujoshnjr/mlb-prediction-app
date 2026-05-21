import csv
import os
import json
import requests
import time

HISTORY_FILE = "data/historical_predictions.csv"
LAST_GAME_FILE = "data/team_last_game.json"

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

def fetch_closing_odds(game_id):
    """
    获取收盘赔率（未来可在此接入 API）
    目前暂无法自动获取，返回 None
    """
    # 示例：未来可通过 Odds-API.io 的历史端点获取
    # url = f"https://api.odds-api.io/v3/historical?event_id={game_id}&apiKey=..."
    # resp = requests.get(url, timeout=10)
    # ...
    return None

def update_results():
    if not os.path.exists(HISTORY_FILE):
        print("历史预测文件不存在，无需更新")
        return

    # 加载休息天数缓存
    if os.path.exists(LAST_GAME_FILE):
        with open(LAST_GAME_FILE, 'r') as f:
            last_game_dict = json.load(f)
    else:
        last_game_dict = {}

    rows = []
    updated_count = 0

    with open(HISTORY_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # 修复字段名中可能的 None 问题
        fieldnames = [fn for fn in reader.fieldnames if fn is not None]
        for row in reader:
            clean_row = {k: v for k, v in row.items() if k in fieldnames}

            # 1. 更新比赛结果
            if clean_row.get("home_win", "").strip() == "" and clean_row.get("game_id", "").strip():
                result = fetch_game_result(clean_row["game_id"])
                if result is not None:
                    clean_row["home_win"] = str(result)
                    updated_count += 1
                    print(f"✅ {clean_row['home_team']} vs {clean_row['away_team']}: home_win={result}")

                    # 更新休息天数缓存
                    home_team = clean_row.get("home_team", "").strip()
                    away_team = clean_row.get("away_team", "").strip()
                    game_date = clean_row.get("game_date", "").strip()
                    if home_team and game_date:
                        last_game_dict[home_team] = game_date
                    if away_team and game_date:
                        last_game_dict[away_team] = game_date
                else:
                    print(f"⏳ {clean_row['home_team']} vs {clean_row['away_team']}: 比赛未结束或数据不可用")
                time.sleep(0.5)

            # 2. 尝试获取收盘赔率（如果 closing_odds 仍为空且 home_odds 已存在）
            if "closing_odds" in clean_row and clean_row["closing_odds"].strip() == "":
                closing = fetch_closing_odds(clean_row["game_id"])
                if closing is not None:
                    clean_row["closing_odds"] = str(closing)

            rows.append(clean_row)

    # 写回历史文件
    with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 保存休息天数缓存
    with open(LAST_GAME_FILE, 'w') as f:
        json.dump(last_game_dict, f, indent=2)

    print(f"结果更新完成，本次更新 {updated_count} 场比赛")

if __name__ == "__main__":
    update_results()
