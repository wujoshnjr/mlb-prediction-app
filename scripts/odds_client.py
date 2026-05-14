"""
Odds-API.io 客户端（使用 requests 直接调用 REST API）
"""
import requests
import pandas as pd
import os

def fetch_odds(api_key: str = None, date_str: str = None) -> pd.DataFrame:
    """
    获取 MLB 赛前赔率，返回 DataFrame
    """
    if not api_key:
        api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("Odds API key missing")
        return pd.DataFrame()

    try:
        # 1. 获取体育列表，找到 MLB 的 key
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
            print("MLB key not found in Odds API")
            return pd.DataFrame()

        # 2. 获取赔率数据
        odds_url = f"https://api.odds-api.io/v4/sports/{baseball_key}/odds"
        params = {
            "regions": "us",
            "markets": "h2h",       # 胜平负/单挑盘
            "oddsFormat": "decimal"
        }
        odds_resp = requests.get(odds_url, headers=headers, params=params, timeout=30)
        odds_resp.raise_for_status()
        odds_data = odds_resp.json()

        # 3. 展开赔率数据为 DataFrame
        rows = []
        for game in odds_data.get("data", []):
            home_team = game.get("home_team")
            away_team = game.get("away_team")
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        rows.append({
                            "sport": baseball_key,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": bookmaker.get("title"),
                            "bet_type": market.get("key"),
                            "team": outcome.get("name"),
                            "odds": outcome.get("price")
                        })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Odds API fetch error: {e}")
        return pd.DataFrame()
