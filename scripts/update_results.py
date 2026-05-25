# scripts/update_results.py
import csv, os, json, requests, time

try:
    from scripts.database import update_game_result, get_connection
    DB_AVAILABLE = True
except:
    DB_AVAILABLE = False

HISTORY_FILE = "data/historical_predictions.csv"
LAST_GAME_FILE = "data/team_last_game.json"
NEW_FINAL_RESULTS_FILE = "data/new_final_results.json"

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
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        home_runs = data.get("liveData",{}).get("linescore",{}).get("teams",{}).get("home",{}).get("runs")
        away_runs = data.get("liveData",{}).get("linescore",{}).get("teams",{}).get("away",{}).get("runs")
        if home_runs is not None and away_runs is not None:
            return 1 if home_runs > away_runs else 0, home_runs, away_runs
    except:
        pass
    return None

def update_results():
    if not os.path.exists(HISTORY_FILE) and not DB_AVAILABLE:
        print("历史数据源不存在")
        return

    if os.path.exists(LAST_GAME_FILE):
        with open(LAST_GAME_FILE, 'r') as f:
            last_game_dict = json.load(f)
    else:
        last_game_dict = {}

    updated_count = 0
    new_final_games = []

    # DB 部分略，仅示范 CSV 更新，实际完整代码可在此基础上加回 DB
    if os.path.exists(HISTORY_FILE):
        rows = []
        csv_updated = 0
        fieldnames = []
        with open(HISTORY_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = [fn for fn in reader.fieldnames if fn is not None]
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
                        new_final_games.append({
                            "game_id": game_id_val.strip(),
                            "game_date": game_date,
                            "home_team": TEAM_NAME_MAP.get(home_team, home_team),
                            "away_team": TEAM_NAME_MAP.get(away_team, away_team),
                            "home_score": home_score,
                            "away_score": away_score
                        })
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
            print(f"CSV 更新完成，更新 {csv_updated} 场")

    # 去重
    seen = set()
    unique_final_games = []
    for g in new_final_games:
        gid = g['game_id']
        if gid not in seen:
            seen.add(gid)
            unique_final_games.append(g)
    with open(NEW_FINAL_RESULTS_FILE, 'w') as f:
        json.dump(unique_final_games, f, indent=2)
    print(f"已保存 {len(unique_final_games)} 场新比赛结果到 {NEW_FINAL_RESULTS_FILE}")

    with open(LAST_GAME_FILE, 'w') as f:
        json.dump(last_game_dict, f, indent=2)

    print(f"结果更新完成，本次更新 {updated_count} 场比赛")

if __name__ == "__main__":
    update_results()
