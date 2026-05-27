# scripts/update_results.py
import csv, os, json, requests, time

try:
    from scripts.database import update_game_result, get_connection
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

try:
    from scripts.snapshot_store import (
        SNAPSHOT_STORE_FILE,
        read_snapshot_rows,
        settle_snapshots,
    )
    SNAPSHOT_AVAILABLE = True
except Exception as exc:
    SNAPSHOT_STORE_FILE = "data/prediction_snapshots.csv"
    SNAPSHOT_AVAILABLE = False
    print(f"Snapshot settlement import failed: {exc}")

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
        status = data.get("gameData",{}).get("status",{})
        abstract_state = status.get("abstractGameState","")
        if abstract_state != "Final":
            print(f"⏳ Game {game_id} 状态 {abstract_state}，跳过")
            return None
        home_runs = data.get("liveData",{}).get("linescore",{}).get("teams",{}).get("home",{}).get("runs")
        away_runs = data.get("liveData",{}).get("linescore",{}).get("teams",{}).get("away",{}).get("runs")
        if home_runs is not None and away_runs is not None:
            return 1 if home_runs > away_runs else 0, home_runs, away_runs
    except Exception as e:
        print(f"获取 game {game_id} 失败: {e}")
    return None

def update_results():
    has_legacy_source = os.path.exists(HISTORY_FILE) or DB_AVAILABLE
    has_snapshot_source = (
        SNAPSHOT_AVAILABLE
        and os.path.exists(SNAPSHOT_STORE_FILE)
    )

    if not has_legacy_source and not has_snapshot_source:
        print("No result source found.")
        return

    if os.path.exists(LAST_GAME_FILE):
        with open(LAST_GAME_FILE) as f:
            last_game_dict = json.load(f)
    else:
        last_game_dict = {}

    updated_count = 0
    new_final_games = []

    # 数据库更新
    if DB_AVAILABLE:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT game_id, home_team, away_team, game_date FROM predictions WHERE home_win IS NULL")
            pending = cursor.fetchall()
            conn.close()

            for row in pending:
                game_id = row["game_id"]
                if not game_id: continue
                result = fetch_game_result(game_id)
                if result:
                    home_win, home_score, away_score = result
                    update_game_result(game_id, home_win)
                    updated_count += 1
                    print(f"✅ {row['home_team']} vs {row['away_team']}: {home_win}")
                    last_game_dict[row['home_team']] = row['game_date']
                    last_game_dict[row['away_team']] = row['game_date']
                    new_final_games.append({
                        "game_id": game_id,
                        "game_date": row['game_date'],
                        "home_team": TEAM_NAME_MAP.get(row['home_team'], row['home_team']),
                        "away_team": TEAM_NAME_MAP.get(row['away_team'], row['away_team']),
                        "home_score": home_score,
                        "away_score": away_score
                    })
                time.sleep(0.5)
        except Exception as e:
            print(f"数据库更新失败: {e}")

    # CSV 更新
    if os.path.exists(HISTORY_FILE):
        rows = []
        csv_updated = 0
        fieldnames = []
        with open(HISTORY_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = [fn for fn in reader.fieldnames if fn is not None]
            if 'home_score' not in fieldnames: fieldnames.append('home_score')
            if 'away_score' not in fieldnames: fieldnames.append('away_score')
            for row in reader:
                clean_row = {k: v for k, v in row.items() if k in fieldnames}
                if clean_row.get("home_win","").strip() == "" and clean_row.get("game_id","").strip():
                    result = fetch_game_result(clean_row["game_id"].strip())
                    if result:
                        home_win, home_score, away_score = result
                        clean_row["home_win"] = str(home_win)
                        clean_row["home_score"] = str(home_score)
                        clean_row["away_score"] = str(away_score)
                        csv_updated += 1
                        home_team = clean_row.get("home_team","").strip()
                        away_team = clean_row.get("away_team","").strip()
                        game_date = clean_row.get("game_date","").strip()
                        if home_team and game_date: last_game_dict[home_team] = game_date
                        if away_team and game_date: last_game_dict[away_team] = game_date
                        new_final_games.append({
                            "game_id": clean_row["game_id"].strip(),
                            "game_date": game_date,
                            "home_team": TEAM_NAME_MAP.get(home_team, home_team),
                            "away_team": TEAM_NAME_MAP.get(away_team, away_team),
                            "home_score": home_score,
                            "away_score": away_score
                        })
                if 'home_score' not in clean_row: clean_row['home_score'] = ''
                if 'away_score' not in clean_row: clean_row['away_score'] = ''
                rows.append(clean_row)
        if csv_updated > 0:
            with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"CSV 更新 {csv_updated} 场")

    # Collect final results for clean snapshots that are not present
    # in the legacy history file.
    if SNAPSHOT_AVAILABLE and os.path.exists(SNAPSHOT_STORE_FILE):
        try:
            pending_snapshots = read_snapshot_rows(
                valid_only=True,
                settled_only=False,
            )

            if not pending_snapshots.empty and "home_win" in pending_snapshots.columns:
                pending_snapshots = pending_snapshots[
                    pending_snapshots["home_win"].isna()
                    | (
                        pending_snapshots["home_win"]
                        .astype(str)
                        .str.strip()
                        == ""
                    )
                ]

            processed_ids = {
                str(game.get("game_id", "")).strip()
                for game in new_final_games
            }

            for _, snapshot in pending_snapshots.iterrows():
                game_id = str(snapshot.get("game_id", "")).strip()
                if not game_id or game_id in processed_ids:
                    continue

                result = fetch_game_result(game_id)
                if not result:
                    time.sleep(0.5)
                    continue

                home_win, home_score, away_score = result
                game_date = str(snapshot.get("game_date", "") or "").strip()
                home_team = str(snapshot.get("home_team", "") or "").strip()
                away_team = str(snapshot.get("away_team", "") or "").strip()

                new_final_games.append({
                    "game_id": game_id,
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_win": home_win,
                    "home_score": home_score,
                    "away_score": away_score,
                })
                processed_ids.add(game_id)

                if home_team and game_date:
                    last_game_dict[home_team] = game_date
                if away_team and game_date:
                    last_game_dict[away_team] = game_date

                time.sleep(0.5)

        except Exception as exc:
            print(f"Snapshot pending result collection failed: {exc}")
            
    # Deduplicate finalized games before writing outputs and settling snapshots.
    seen = set()
    unique_final = []
    for game in new_final_games:
        game_id = str(game.get("game_id", "")).strip()
        if game_id and game_id not in seen:
            seen.add(game_id)
            unique_final.append(game)

    snapshot_settlement_summary = {
        "updated": 0,
        "unmatched": 0,
        "stored_rows": 0,
    }
    if SNAPSHOT_AVAILABLE and os.path.exists(SNAPSHOT_STORE_FILE):
        try:
            snapshot_settlement_summary = settle_snapshots(unique_final)
            print(
                "Snapshot settlement: "
                f"updated={snapshot_settlement_summary.get('updated', 0)}, "
                f"unmatched={snapshot_settlement_summary.get('unmatched', 0)}"
            )
        except Exception as exc:
            print(f"Snapshot settlement failed: {exc}")

    with open(NEW_FINAL_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(unique_final, f, indent=2)
    print(f"Saved {len(unique_final)} newly finalized games.")

    with open(LAST_GAME_FILE, 'w') as f:
        json.dump(last_game_dict, f, indent=2)

    print(f"本次更新 {updated_count} 场比赛")

if __name__ == "__main__":
    update_results()
