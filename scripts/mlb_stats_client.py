"""
MLB Stats API 客户端（增强版）
从 /v1/schedule 抓取当日完整赛程，包含比赛时间、球场、两队先发投手等
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
            # 比赛时间（UTC，可转本地）
            game_date = game.get("gameDate")  # ISO 8601 格式
            # 状态
            status = game.get("status", {}).get("abstractGameState", "Unknown")
            # 球队
            teams = game.get("teams", {})
            home_team = teams.get("home", {}).get("team", {})
            away_team = teams.get("away", {}).get("team", {})
            # 球场
            venue = game.get("venue", {}).get("name", "")
            # 先发投手（通过 hydrate=probablePitcher）
            home_pitcher_id = None
            away_pitcher_id = None
            if teams.get("home", {}).get("probablePitcher"):
                home_pitcher_id = teams["home"]["probablePitcher"]["id"]
            if teams.get("away", {}).get("probablePitcher"):
                away_pitcher_id = teams["away"]["probablePitcher"]["id"]

            games.append({
                "game_id": game_id,
                "game_date": game_date,          # ISO 8601，如 "2026-05-18T23:10:00Z"
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
