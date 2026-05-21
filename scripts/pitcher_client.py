import requests
import pandas as pd
from datetime import datetime

def fetch_probable_pitchers(date_str: str = None, errors: list = None) -> pd.DataFrame:
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitcher",
        "gameTypes": "R"
    }
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
            home_pitcher_id = None
            away_pitcher_id = None
            home_team = game.get("teams", {}).get("home", {})
            away_team = game.get("teams", {}).get("away", {})
            if home_team.get("probablePitcher"):
                home_pitcher_id = home_team["probablePitcher"]["id"]
            if away_team.get("probablePitcher"):
                away_pitcher_id = away_team["probablePitcher"]["id"]
            if not home_pitcher_id or not away_pitcher_id:
                continue

            pitcher_ids = [home_pitcher_id, away_pitcher_id]
            pitcher_info = {}
            for pid in pitcher_ids:
                # 获取投手个人信息（包含投球手）
                person_url = f"https://statsapi.mlb.com/api/v1/people/{pid}"
                pitch_hand = "R"
                try:
                    person_resp = requests.get(person_url, timeout=5)
                    person_resp.raise_for_status()
                    person_data = person_resp.json().get("people", [{}])[0]
                    pitch_hand = person_data.get("pitchHand", {}).get("code", "R")
                except:
                    pass

                # 获取赛季统计
                stats_url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                stats_params = {"stats": "season", "season": 2026, "gameType": "R"}
                try:
                    stats_resp = requests.get(stats_url, params=stats_params, timeout=10)
                    stats_resp.raise_for_status()
                    splits = stats_resp.json().get("stats", [])
                    stat = splits[0].get("splits", [{}])[0].get("stat", {}) if splits else {}
                except:
                    stat = {}

                pitcher_info[pid] = {
                    "era": stat.get("era"),
                    "whip": stat.get("whip"),
                    "k_per_9": stat.get("strikeoutsPer9Inn"),
                    "bb_per_9": stat.get("walksPer9Inn"),
                    "fip": stat.get("fip"),
                    "innings_pitched": stat.get("inningsPitched"),
                    "pitch_hand": pitch_hand           # 新增
                }

            games.append({
                "game_id": game_id,
                "home_pitcher_id": home_pitcher_id,
                "away_pitcher_id": away_pitcher_id,
                "home_era": pitcher_info.get(home_pitcher_id, {}).get("era"),
                "home_fip": pitcher_info.get(home_pitcher_id, {}).get("fip"),
                "home_whip": pitcher_info.get(home_pitcher_id, {}).get("whip"),
                "home_k9": pitcher_info.get(home_pitcher_id, {}).get("k_per_9"),
                "home_bb9": pitcher_info.get(home_pitcher_id, {}).get("bb_per_9"),
                "home_pitch_hand": pitcher_info.get(home_pitcher_id, {}).get("pitch_hand", "R"),
                "away_era": pitcher_info.get(away_pitcher_id, {}).get("era"),
                "away_fip": pitcher_info.get(away_pitcher_id, {}).get("fip"),
                "away_whip": pitcher_info.get(away_pitcher_id, {}).get("whip"),
                "away_k9": pitcher_info.get(away_pitcher_id, {}).get("k_per_9"),
                "away_bb9": pitcher_info.get(away_pitcher_id, {}).get("bb_per_9"),
                "away_pitch_hand": pitcher_info.get(away_pitcher_id, {}).get("pitch_hand", "R"),
            })
    return pd.DataFrame(games)
