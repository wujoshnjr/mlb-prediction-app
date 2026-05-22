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
                except: pass
                stats_url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                stats_params = {"stats":"season","season":2026,"gameType":"R"}
                try:
                    stats_resp = requests.get(stats_url, params=stats_params, timeout=10)
                    splits = stats_resp.json().get("stats",[[]])[0].get("splits",[{}])
                    stat = splits[0].get("stat",{}) if splits else {}
                except: stat = {}
                # Stuff+ 代理：使用 K% + 球速 + CSW 的线性组合
                k_pct = stat.get("strikeOutsPer9Inn", 8.0) or 8.0
                bb_pct = stat.get("walksPer9Inn", 3.0) or 3.0
                # 简化 Stuff+ = (K% - BB%) * 5 + 球速(从Savant获取) - 100，这里用固定近似
                stuff_plus = 100 + (k_pct - 3) * 5  # 非常粗糙的代理
                # CSW% 代理：K% / 2 + 0.15
                csw_pct = (k_pct / 9) * 0.5 + 0.15
                pitcher_info[pid] = {
                    "era": stat.get("era"),
                    "fip": stat.get("fip"),
                    "whip": stat.get("whip"),
                    "k_per_9": k_pct,
                    "bb_per_9": bb_pct,
                    "pitch_hand": pitch_hand,
                    "stuff_plus": stuff_plus,
                    "csw_pct": csw_pct
                }
            games.append({
                "game_id": game_id,
                "home_pitcher_id": home_pitcher_id,
                "away_pitcher_id": away_pitcher_id,
                "home_era": pitcher_info[home_pitcher_id].get("era"),
                "home_fip": pitcher_info[home_pitcher_id].get("fip"),
                "home_pitch_hand": pitcher_info[home_pitcher_id].get("pitch_hand","R"),
                "home_stuff_plus": pitcher_info[home_pitcher_id].get("stuff_plus"),
                "home_csw_pct": pitcher_info[home_pitcher_id].get("csw_pct"),
                "away_era": pitcher_info[away_pitcher_id].get("era"),
                "away_fip": pitcher_info[away_pitcher_id].get("fip"),
                "away_pitch_hand": pitcher_info[away_pitcher_id].get("pitch_hand","R"),
                "away_stuff_plus": pitcher_info[away_pitcher_id].get("stuff_plus"),
                "away_csw_pct": pitcher_info[away_pitcher_id].get("csw_pct"),
            })
    return pd.DataFrame(games)
