import csv
import os
import json
import requests
import time

# 导入数据库模块
try:
    from scripts.database import update_game_result, get_connection
    DB_AVAILABLE = True
except:
    DB_AVAILABLE = False

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
    except:
        pass
    return None

def update_results():
    if not os.path.exists(HISTORY_FILE) and not DB_AVAILABLE:
        print("历史数据源不存在")
        return

    # 加载休息天数缓存
    if os.path.exists(LAST_GAME_FILE):
        with open(LAST_GAME_FILE, 'r') as f:
            last_game_dict = json.load(f)
    else:
        last_game_dict = {}

    updated_count = 0

    # 优先使用数据库更新
    if DB_AVAILABLE:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT game_id, home_team, away_team, game_date FROM predictions WHERE home_win IS NULL")
            pending = cursor.fetchall()
            conn.close()

            for row in pending:
                game_id = row["game_id"]
                if not game_id:
                    continue
                home_team = row["home_team"] or ""
                away_team = row["away_team"] or ""
                game_date = row["game_date"] or ""
                result = fetch_game_result(game_id)
                if result is not None:
                    update_game_result(game_id, result)
                    updated_count += 1
                    print(f"✅ {home_team} vs {away_team}: home_win={result}")
                    # 更新休息天数缓存
                    if home_team and game_date:
                        last_game_dict[home_team] = game_date
                    if away_team and game_date:
                        last_game_dict[away_team] = game_date
                else:
                    print(f"⏳ {home_team} vs {away_team}: 未结束")
                time.sleep(0.5)
        except Exception as e:
            print(f"数据库更新失败: {e}，回退到 CSV")

    # 同时更新 CSV（如果存在）
    if os.path.exists(HISTORY_FILE):
        rows = []
        csv_updated = 0
        with open(HISTORY_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = [fn for fn in reader.fieldnames if fn is not None]
            for row in reader:
                clean_row = {k: v for k, v in row.items() if k in fieldnames}
                # 安全获取 home_win 和 game_id，处理 None 值
                home_win_val = clean_row.get("home_win") or ""
                game_id_val = clean_row.get("game_id") or ""
                if home_win_val.strip() == "" and game_id_val.strip():
                    result = fetch_game_result(game_id_val.strip())
                    if result is not None:
                        clean_row["home_win"] = str(result)
                        csv_updated += 1
                        home_team = (clean_row.get("home_team") or "").strip()
                        away_team = (clean_row.get("away_team") or "").strip()
                        game_date = (clean_row.get("game_date") or "").strip()
                        if home_team and game_date:
                            last_game_dict[home_team] = game_date
                        if away_team and game_date:
                            last_game_dict[away_team] = game_date
                rows.append(clean_row)
        if csv_updated > 0:
            with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"CSV 更新完成，更新 {csv_updated} 场")

    # 保存休息天数缓存
    with open(LAST_GAME_FILE, 'w') as f:
        json.dump(last_game_dict, f, indent=2)

    print(f"结果更新完成，本次更新 {updated_count} 场比赛")

if __name__ == "__main__":
    update_results()
