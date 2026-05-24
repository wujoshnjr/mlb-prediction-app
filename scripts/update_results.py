# scripts/update_results.py
import csv
import os
import json
import requests
import time

try:
    from scripts.database import update_game_result, get_connection
    DB_AVAILABLE = True
except:
    DB_AVAILABLE = False

HISTORY_FILE = "data/historical_predictions.csv"
LAST_GAME_FILE = "data/team_last_game.json"
NEW_FINAL_RESULTS_FILE = "data/new_final_results.json"

# 統一的隊名映射（與 prediction.py 保持一致）
TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "D-backs", "Diamondbacks": "D-backs",
    "Atlanta Braves": "Braves", "Baltimore Orioles": "Orioles",
    "Boston Red Sox": "Red Sox", "Chicago Cubs": "Cubs",
    "Chicago White Sox": "White Sox", "Cincinnati Reds": "Reds",
    "Cleveland Guardians": "Guardians", "Colorado Rockies": "Rockies",
    "Detroit Tigers": "Tigers", "Houston Astros": "Astros",
    "Kansas City Royals": "Royals", "Los Angeles Angels": "Angels",
    "Los Angeles Dodgers": "Dodgers", "Miami Marlins": "Marlins",
    "Milwaukee Brewers": "Brewers", "Minnesota Twins": "Twins",
    "New York Mets": "Mets", "New York Yankees": "Yankees",
    "Oakland Athletics": "Athletics", "Athletics": "Athletics",
    "Philadelphia Phillies": "Phillies", "Pittsburgh Pirates": "Pirates",
    "San Diego Padres": "Padres", "San Francisco Giants": "Giants",
    "Seattle Mariners": "Mariners", "St. Louis Cardinals": "Cardinals",
    "Tampa Bay Rays": "Rays", "Texas Rangers": "Rangers",
    "Toronto Blue Jays": "Blue Jays", "Washington Nationals": "Nationals"
}

def fetch_game_result(game_id):
    """
    獲取比賽結果，返回 (home_win, home_score, away_score) 或 None。
    home_win: 1 主隊勝, 0 客隊勝
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        home_runs = data.get("liveData", {}).get("linescore", {}).get("teams", {}).get("home", {}).get("runs")
        away_runs = data.get("liveData", {}).get("linescore", {}).get("teams", {}).get("away", {}).get("runs")
        if home_runs is not None and away_runs is not None:
            home_win = 1 if home_runs > away_runs else 0
            return home_win, home_runs, away_runs
    except Exception as e:
        print(f"獲取比賽結果失敗 (game_id={game_id}): {e}")
    return None

def update_results():
    if not os.path.exists(HISTORY_FILE) and not DB_AVAILABLE:
        print("歷史數據源不存在")
        return

    # 加載休息天數緩存
    if os.path.exists(LAST_GAME_FILE):
        with open(LAST_GAME_FILE, 'r') as f:
            last_game_dict = json.load(f)
    else:
        last_game_dict = {}

    updated_count = 0
    new_final_games = []  # 收集本次更新的 final 比賽

    # 優先使用數據庫更新
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
                    home_win, home_score, away_score = result
                    update_game_result(game_id, home_win)
                    updated_count += 1
                    print(f"✅ {home_team} vs {away_team}: home_win={home_win} ({home_score}-{away_score})")
                    # 更新休息天數緩存
                    if home_team and game_date:
                        last_game_dict[home_team] = game_date
                    if away_team and game_date:
                        last_game_dict[away_team] = game_date
                    # 收集 final 比賽
                    new_final_games.append({
                        "game_id": game_id,
                        "game_date": game_date,
                        "home_team": TEAM_NAME_MAP.get(home_team, home_team),
                        "away_team": TEAM_NAME_MAP.get(away_team, away_team),
                        "home_score": home_score,
                        "away_score": away_score
                    })
                else:
                    print(f"⏳ {home_team} vs {away_team}: 未結束")
                time.sleep(0.5)
        except Exception as e:
            print(f"數據庫更新失敗: {e}，回退到 CSV")

    # 同時更新 CSV（如果存在）
    if os.path.exists(HISTORY_FILE):
        rows = []
        csv_updated = 0
        fieldnames = []
        with open(HISTORY_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = [fn for fn in reader.fieldnames if fn is not None]
            # 確保 CSV 有 home_score, away_score 欄位，若無則添加
            if 'home_score' not in fieldnames:
                fieldnames.append('home_score')
            if 'away_score' not in fieldnames:
                fieldnames.append('away_score')
            for row in reader:
                clean_row = {k: v for k, v in row.items() if k in fieldnames}
                home_win_val = clean_row.get("home_win") or ""
                game_id_val = clean_row.get("game_id") or ""
                if home_win_val.strip() == "" and game_id_val.strip():
                    result = fetch_game_result(game_id_val.strip())
                    if result is not None:
                        home_win, home_score, away_score = result
                        clean_row["home_win"] = str(home_win)
                        clean_row["home_score"] = str(home_score)
                        clean_row["away_score"] = str(away_score)
                        csv_updated += 1
                        home_team = (clean_row.get("home_team") or "").strip()
                        away_team = (clean_row.get("away_team") or "").strip()
                        game_date = (clean_row.get("game_date") or "").strip()
                        if home_team and game_date:
                            last_game_dict[home_team] = game_date
                        if away_team and game_date:
                            last_game_dict[away_team] = game_date
                        # 收集 final 比賽
                        new_final_games.append({
                            "game_id": game_id_val.strip(),
                            "game_date": game_date,
                            "home_team": TEAM_NAME_MAP.get(home_team, home_team),
                            "away_team": TEAM_NAME_MAP.get(away_team, away_team),
                            "home_score": home_score,
                            "away_score": away_score
                        })
                # 確保現有行也有比分欄位（若缺失則留空）
                if 'home_score' not in clean_row:
                    clean_row['home_score'] = ''
                if 'away_score' not in clean_row:
                    clean_row['away_score'] = ''
                rows.append(clean_row)
        if csv_updated > 0:
            with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"CSV 更新完成，更新 {csv_updated} 場")

    # 保存新的 final results 供 rating_updater 使用
    if new_final_games:
        with open(NEW_FINAL_RESULTS_FILE, 'w') as f:
            json.dump(new_final_games, f, indent=2)
        print(f"已保存 {len(new_final_games)} 場新比賽結果到 {NEW_FINAL_RESULTS_FILE}")
    else:
        # 若無新比賽，仍寫入空列表避免 rating_updater 報錯
        with open(NEW_FINAL_RESULTS_FILE, 'w') as f:
            json.dump([], f)
        print("沒有新的 final 比賽")

    # 保存休息天數緩存
    with open(LAST_GAME_FILE, 'w') as f:
        json.dump(last_game_dict, f, indent=2)

    print(f"結果更新完成，本次更新 {updated_count} 場比賽")

if __name__ == "__main__":
    update_results()
