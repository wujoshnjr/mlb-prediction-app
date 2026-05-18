import requests
import pandas as pd

BASE_URL = "https://statsapi.mlb.com/api/v1"

def fetch_mlb_statsapi(date_str: str = None, errors: list = None) -> pd.DataFrame:
    endpoint = f"{BASE_URL}/schedule"
    params = {
        "sportId": 1,
        "gameTypes": "R",
        "hydrate": "team"
    }
    if date_str:
        params["date"] = date_str
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
            teams = game.get("teams", {})
            home = teams.get("home", {}).get("team", {})
            away = teams.get("away", {}).get("team", {})
            games.append({
                "game_id": game.get("gamePk"),
                "home": home.get("name", "Unknown") if home else "Unknown",
                "away": away.get("name", "Unknown") if away else "Unknown",
                "status": game.get("status", {}).get("abstractGameState", "Unknown")
            })
    return pd.DataFrame(games)
