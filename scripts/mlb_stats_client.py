"""
MLB Stats API 客户端
直接使用 requests 获取实时赛程与比分（无需第三方 MLB 套件）
"""
import requests
import pandas as pd

BASE_URL = "https://statsapi.mlb.com/api/v1"

def fetch_mlb_statsapi(date_str: str = None) -> pd.DataFrame:
    """
    获取指定日期的赛程，若无日期则获取当日赛程
    返回包含 game_id, home, away, status 的 DataFrame
    """
    endpoint = f"{BASE_URL}/schedule"
    params = {
        "sportId": 1,          # MLB
        "date": date_str if date_str else None,
        "gameTypes": "R",      # 常规赛
        "hydrate": "team"
    }
    # 如果没有指定日期，就不传 date 参数，API 会返回当日赛程
    if not date_str:
        params.pop("date", None)

    try:
        resp = requests.get(endpoint, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"MLB Stats API request failed: {e}")
        return pd.DataFrame()

    games = []
    for date_info in data.get("dates", []):
        for game in date_info.get("games", []):
            teams = game.get("teams", {})
            games.append({
                "game_id": game.get("gamePk"),
                "home": teams.get("home", {}).get("team", {}).get("name", ""),
                "away": teams.get("away", {}).get("team", {}).get("name", ""),
                "status": game.get("status", {}).get("abstractGameState", "")
            })

    return pd.DataFrame(games)
