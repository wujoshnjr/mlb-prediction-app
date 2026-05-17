"""
Odds-API.io 客户端（修正端点 + 健壮的数据解析）
"""
import requests
import pandas as pd
import os

def fetch_odds(api_key: str = None, date_str: str = None, errors: list = None) -> pd.DataFrame:
    if not api_key:
        raw = os.getenv("ODDS_API_KEY", "") or ""
        api_key = raw.strip().replace("\n", "").replace("\r", "")
    if not api_key:
        if errors is not None:
            errors.append("Odds API key missing")
        return pd.DataFrame()

    try:
        # 获取体育列表
        sports_url = "https://api.the-odds-api.com/v4/sports"
        params = {"apiKey": api_key}
        sports_resp = requests.get(sports_url, params=params, timeout=15)
        sports_resp.raise_for_status()
        sports_data = sports_resp.json()

        # 健壮解析：兼容列表和字典两种返回格式
        if isinstance(sports_data, list):
            sport_list = sports_data
        else:
            sport_list = sports_data.get("data", [])

        # 寻找 MLB 的 sport key
        baseball_key = None
        for sport in sport_list:
            if sport.get("group") == "Baseball" and "MLB" in sport.get("title", ""):
                baseball_key = sport["key"]
                break

        if not baseball_key:
            if errors is not None:
                errors.append("MLB key not found in Odds API")
            return pd.DataFrame()

        # 获取赔率
        odds_url = f"https://api.the-odds-api.com/v4/sports/{baseball_key}/odds"
        odds_params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "bookmakers": "bet365,draftkings"
        }
        odds_resp = requests.get(odds_url, params=odds_params, timeout=30)

        if odds_resp.status_code == 401:
            if errors is not None:
                errors.append("Odds API 401: API key invalid. Get a free key at the-odds-api.com")
            return pd.DataFrame()

        odds_resp.raise_for_status()
        odds_data = odds_resp.json()

        # 健壮解析赔率数据
        if isinstance(odds_data, list):
            games = odds_data
        else:
            games = odds_data.get("data", [])

        # 展开为扁平表格
        rows = []
        for game in games:
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
        if errors is not None:
            errors.append(f"Odds API HTTP error: {e}")
        return pd.DataFrame()
    except Exception as e:
        if errors is not None:
            errors.append(f"Odds API fetch error: {e}")
        return pd.DataFrame()
