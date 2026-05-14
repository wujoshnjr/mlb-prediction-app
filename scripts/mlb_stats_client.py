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
        msg = f"MLB Stats API error: {e}"
        if errors is not None:
            errors.append(msg)
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
