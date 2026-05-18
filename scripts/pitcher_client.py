"""
先发投手数据客户端
"""
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
            home_pitcher = None
            away_pitcher = None
            home_team = game.get("teams", {}).get("home", {})
            away_team = game.get("teams", {}).get("away", {})
            if home_team.get("probablePitcher"):
                home_pitcher = home_team["probablePitcher"]["id"]
            if away_team.get("probablePitcher"):
                away_pitcher = away_team["probablePitcher"]["id"]
            if not home_pitcher or not away_pitcher:
                continue
            pitcher_ids = [home_pitcher, away_pitcher]
            pitcher_stats = {}
            for pid in pitcher_ids:
                try:
                    stats_url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                    stats_params = {"stats": "season", "season": 2026, "gameType": "R"}
                    stats_resp = requests.get(stats_url, params=stats_params, timeout=10)
                    stats_resp.raise_for_status()
                    splits = stats_resp.json().get("stats", [])
                    if splits:
                        stat = splits[0].get("splits", [{}])[0].get("stat", {})
                        pitcher_stats[pid] = {
                            "era": stat.get("era"),
                            "whip": stat.get("whip"),
                            "k_per_9": stat.get("strikeoutsPer9Inn"),
                            "bb_per_9": stat.get("walksPer9Inn"),
                            "fip": stat.get("fip"),
                            "innings_pitched": stat.get("inningsPitched")
                        }
                except:
                    pitcher_stats[pid] = {}
            games.append({
                "game_id": game_id,
                "home_pitcher_id": home_pitcher,
                "away_pitcher_id": away_pitcher,
                "home_era": pitcher_stats.get(home_pitcher, {}).get("era"),
                "home_whip": pitcher_stats.get(home_pitcher, {}).get("whip"),
                "home_k9": pitcher_stats.get(home_pitcher, {}).get("k_per_9"),
                "home_bb9": pitcher_stats.get(home_pitcher, {}).get("bb_per_9"),
                "home_fip": pitcher_stats.get(home_pitcher, {}).get("fip"),
                "away_era": pitcher_stats.get(away_pitcher, {}).get("era"),
                "away_whip": pitcher_stats.get(away_pitcher, {}).get("whip"),
                "away_k9": pitcher_stats.get(away_pitcher, {}).get("k_per_9"),
                "away_bb9": pitcher_stats.get(away_pitcher, {}).get("bb_per_9"),
                "away_fip": pitcher_stats.get(away_pitcher, {}).get("fip"),
            })
    return pd.DataFrame(games)
