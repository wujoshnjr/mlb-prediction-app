"""
MLB Stats API 客户端（增强版）
从 /v1/schedule 抓取当日完整赛程，包含比赛时间、球场、先发投手 ID
"""
import requests
import pandas as pd
from datetime import datetime

BASE_URL = "https://statsapi.mlb.com/api/v1"

def fetch_mlb_statsapi(date_str: str = None, errors: list = None) -> pd.DataFrame:
    endpoint = f"{BASE_URL}/schedule"
    params = {
        "sportId": 1,
        "date": date_str if date_str else datetime.now().strftime("%Y-%m-%d"),
        "hydrate": "team,probablePitcher,venue",
        "gameTypes": "R"
    }
    try:
        resp = requests.get(endpoint, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        if errors is not None:
            errors.append(f"MLB Stats API error: {e}")
        return pd.DataFrame()

    games = []
    for date_info in data.get("dates", []):
        for game in date_info.get("games", []):
            game_id = game["gamePk"]
            game_date = game.get("gameDate")
            status = game.get("status", {}).get("abstractGameState", "Unknown")
            teams = game.get("teams", {})
            home_team = teams.get("home", {}).get("team", {})
            away_team = teams.get("away", {}).get("team", {})
            venue = game.get("venue", {}).get("name", "")
            # 先发投手
            home_pitcher_id = None
            away_pitcher_id = None
            if teams.get("home", {}).get("probablePitcher"):
                home_pitcher_id = teams["home"]["probablePitcher"]["id"]
            if teams.get("away", {}).get("probablePitcher"):
                away_pitcher_id = teams["away"]["probablePitcher"]["id"]

            games.append({
                "game_id": game_id,
                "game_date": game_date,
                "status": status,
                "home_team": home_team.get("name", "Unknown"),
                "away_team": away_team.get("name", "Unknown"),
                "home_team_id": home_team.get("id"),
                "away_team_id": away_team.get("id"),
                "venue": venue,
                "home_pitcher_id": home_pitcher_id,
                "away_pitcher_id": away_pitcher_id,
            })
    return pd.DataFrame(games)
