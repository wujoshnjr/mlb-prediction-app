import requests
import pandas as pd
from datetime import datetime

def fetch_probable_pitchers(date_str: str = None, errors: list = None) -> pd.DataFrame:
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId":1,"date":date_str,"hydrate":"probablePitcher","gameTypes":"R"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        if errors is not None:
            errors.append(f"Pitcher fetch error: {e}")
        return pd.DataFrame()

    games = []
    for date_info in data.get("dates", []):
        for game in date_info.get("games", []):
            game_id = game["gamePk"]
            home_pitcher_id = away_pitcher_id = None
            if game["teams"]["home"].get("probablePitcher"):
                home_pitcher_id = game["teams"]["home"]["probablePitcher"]["id"]
            if game["teams"]["away"].get("probablePitcher"):
                away_pitcher_id = game["teams"]["away"]["probablePitcher"]["id"]
            if not home_pitcher_id or not away_pitcher_id:
                continue
            pitcher_info = {}
            for pid in [home_pitcher_id, away_pitcher_id]:
                person_url = f"https://statsapi.mlb.com/api/v1/people/{pid}"
                pitch_hand = "R"
                try:
                    person_resp = requests.get(person_url, timeout=5)
                    person_data = person_resp.json().get("people",[{}])[0]
                    pitch_hand = person_data.get("pitchHand",{}).get("code","R")
                except:
                    pass
                stats_url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                stats_params = {"stats":"season","season":2026,"gameType":"R"}
                try:
                    stats_resp = requests.get(stats_url, params=stats_params, timeout=10)
                    splits = stats_resp.json().get("stats",[[]])[0].get("splits",[{}])
                    stat = splits[0].get("stat",{}) if splits else {}
                except:
                    stat = {}
                # 获取首局数据（byInning=1）
                first_inning_stats = {}
                try:
                    first_params = {"stats":"byInning","season":2026,"gameType":"R","inning":1}
                    first_resp = requests.get(stats_url, params=first_params, timeout=10)
                    first_splits = first_resp.json().get("stats",[[]])[0].get("splits",[{}])
                    first_stat = first_splits[0].get("stat",{}) if first_splits else {}
                    first_inning_stats = first_stat
                except:
                    pass
                pitcher_info[pid] = {
                    "era": stat.get("era"),
                    "fip": stat.get("fip"),
                    "whip": stat.get("whip"),
                    "k_per_9": stat.get("strikeoutsPer9Inn"),
                    "bb_per_9": stat.get("walksPer9Inn"),
                    "pitch_hand": pitch_hand,
                    "first_era": first_inning_stats.get("era")  # 首局ERA
                }
            games.append({
                "game_id": game_id,
                "home_pitcher_id": home_pitcher_id,
                "away_pitcher_id": away_pitcher_id,
                "home_era": pitcher_info[home_pitcher_id].get("era"),
                "home_fip": pitcher_info[home_pitcher_id].get("fip"),
                "home_pitch_hand": pitcher_info[home_pitcher_id].get("pitch_hand","R"),
                "home_first_era": pitcher_info[home_pitcher_id].get("first_era"),
                "away_era": pitcher_info[away_pitcher_id].get("era"),
                "away_fip": pitcher_info[away_pitcher_id].get("fip"),
                "away_pitch_hand": pitcher_info[away_pitcher_id].get("pitch_hand","R"),
                "away_first_era": pitcher_info[away_pitcher_id].get("first_era"),
            })
    return pd.DataFrame(games)
