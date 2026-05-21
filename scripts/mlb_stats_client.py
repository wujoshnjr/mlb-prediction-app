"""
MLB Stats API 客户端（增强版：包含 lineup 捕手信息、日场/夜场）
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
        "hydrate": "team,probablePitcher,venue,lineup",
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

            # 先发捕手（从 lineup 中获取 catcher 位置球员，position code = 2）
            home_catcher_id = None
            away_catcher_id = None
            try:
                home_lineup = teams["home"].get("lineup", {}).get("lineup", [])
                for player in home_lineup:
                    if player.get("position", {}).get("code") == "2":
                        home_catcher_id = player.get("player", {}).get("id")
                        break
            except:
                pass
            try:
                away_lineup = teams["away"].get("lineup", {}).get("lineup", [])
                for player in away_lineup:
                    if player.get("position", {}).get("code") == "2":
                        away_catcher_id = player.get("player", {}).get("id")
                        break
            except:
                pass

            # 日场/夜场
            day_night = game.get("dayNight", "")
            is_day_game = 1 if "day" in str(day_night).lower() else 0

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
                "home_catcher_id": home_catcher_id,
                "away_catcher_id": away_catcher_id,
                "is_day_game": is_day_game,
                "start_time": game.get("gameDate")
            })
    return pd.DataFrame(games)
