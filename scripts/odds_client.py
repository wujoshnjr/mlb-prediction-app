"""
Odds-API.io 客户端（增強錯誤提示）
"""
import requests
import pandas as pd
import os

def fetch_odds(api_key: str = None, date_str: str = None, errors: list = None) -> pd.DataFrame:
    if not api_key:
        api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        msg = "Odds API key missing"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()

    try:
        sports_url = "https://api.odds-api.io/v4/sports"
        headers = {"apikey": api_key}
        sports_resp = requests.get(sports_url, headers=headers, timeout=15)
        sports_resp.raise_for_status()
        sports_data = sports_resp.json()

        baseball_key = None
        for sport in sports_data.get("data", []):
            if sport.get("group") == "Baseball" and "MLB" in sport.get("title", ""):
                baseball_key = sport["key"]
                break

        if not baseball_key:
            msg = "MLB key not found in Odds API"
            if errors is not None:
                errors.append(msg)
            return pd.DataFrame()

        odds_url = f"https://api.odds-api.io/v4/sports/{baseball_key}/odds"
        params = {
            "apikey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "bookmakers": "bet365,draftkings"
        }
        odds_resp = requests.get(odds_url, headers=headers, params=params, timeout=30)

        if odds_resp.status_code == 401:
            msg = "Odds API 401 未授權，請檢查 API Key 是否正確（應以 oddsp- 開頭）"
            if errors is not None:
                errors.append(msg)
            return pd.DataFrame()

        odds_resp.raise_for_status()
        odds_data = odds_resp.json()

        if not odds_data.get("data"):
            msg = "Odds API returned no game data (可能當天無 MLB 比賽，或博彩公司不支援)"
            if errors is not None:
                errors.append(msg)
            return pd.DataFrame()

        rows = []
        for game in odds_data["data"]:
            home_team = game.get("home_team")
            away_team = game.get("away_team")
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        rows.append({
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": bookmaker.get("title"),
                            "bet_type": market.get("key"),
                            "team": outcome.get("name"),
                            "odds": outcome.get("price")
                        })
        return pd.DataFrame(rows)

    except requests.exceptions.HTTPError as e:
        msg = f"Odds API HTTP error: {e}"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()
    except Exception as e:
        msg = f"Odds API fetch error: {e}"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()
