import os, time, requests, pandas as pd
from datetime import datetime, timedelta

DATA_DIR = "data/historical"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_schedule(date_str):
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": date_str, "hydrate": "team,probablePitcher,venue"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except:
        return None

def fetch_game_result(game_id):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        home_runs = data.get("liveData",{}).get("linescore",{}).get("teams",{}).get("home",{}).get("runs")
        away_runs = data.get("liveData",{}).get("linescore",{}).get("teams",{}).get("away",{}).get("runs")
        if home_runs is not None and away_runs is not None:
            return 1 if home_runs > away_runs else 0
    except:
        pass
    return None

def fetch_pitcher_stats(pitcher_id, season):
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
    params = {"stats":"season","season":season,"gameType":"R"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        splits = resp.json().get("stats",[[]])[0].get("splits",[{}])
        return splits[0].get("stat",{}) if splits else {}
    except:
        return {}

def collect_date(date_str, season, last_game_dict=None):
    schedule = fetch_schedule(date_str)
    if not schedule:
        return last_game_dict
    games = []
    for date_info in schedule.get("dates",[]):
        for game in date_info.get("games",[]):
            game_id = game["gamePk"]
            home_team = game["teams"]["home"]["team"]["name"]
            away_team = game["teams"]["away"]["team"]["name"]
            home_pitcher_id = away_pitcher_id = None
            if game["teams"]["home"].get("probablePitcher"):
                home_pitcher_id = game["teams"]["home"]["probablePitcher"]["id"]
            if game["teams"]["away"].get("probablePitcher"):
                away_pitcher_id = game["teams"]["away"]["probablePitcher"]["id"]
            home_win = None
            if game.get("status",{}).get("abstractGameState") == "Final":
                home_win = fetch_game_result(game_id)
            home_pitcher_stats = fetch_pitcher_stats(home_pitcher_id, season) if home_pitcher_id else {}
            away_pitcher_stats = fetch_pitcher_stats(away_pitcher_id, season) if away_pitcher_id else {}
            rest_home = rest_away = 2
            if last_game_dict:
                try:
                    game_dt = datetime.strptime(date_str, "%Y-%m-%d")
                    if home_team in last_game_dict:
                        last = datetime.strptime(last_game_dict[home_team], "%Y-%m-%d")
                        rest_home = max(0, (game_dt - last).days - 1)
                    if away_team in last_game_dict:
                        last = datetime.strptime(last_game_dict[away_team], "%Y-%m-%d")
                        rest_away = max(0, (game_dt - last).days - 1)
                except:
                    pass
            if last_game_dict is not None:
                last_game_dict[home_team] = date_str
                last_game_dict[away_team] = date_str
            games.append({
                "game_id": game_id, "date": date_str, "home_team": home_team, "away_team": away_team,
                "home_win": home_win, "home_era": home_pitcher_stats.get("era"),
                "away_era": away_pitcher_stats.get("era"), "home_fip": home_pitcher_stats.get("fip"),
                "away_fip": away_pitcher_stats.get("fip"), "rest_home": rest_home, "rest_away": rest_away
            })
    df = pd.DataFrame(games)
    df.to_parquet(os.path.join(DATA_DIR, f"{date_str}.parquet"), index=False)
    print(f"Saved {len(games)} games for {date_str}")
    return last_game_dict

def collect_range(start_date, end_date):
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    last_game_dict = {}
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        if not os.path.exists(os.path.join(DATA_DIR, f"{date_str}.parquet")):
            last_game_dict = collect_date(date_str, current.year, last_game_dict)
            time.sleep(0.5)
        current += timedelta(days=1)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        collect_range(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        collect_date(sys.argv[1], datetime.strptime(sys.argv[1], "%Y-%m-%d").year)
