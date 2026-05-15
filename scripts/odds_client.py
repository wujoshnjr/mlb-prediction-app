"""
Odds-API.io 客户端（直接调用 REST API）
"""
import requests
import pandas as pd
import os

def fetch_odds(api_key: str = None, date_str: str = None, errors: list = None) -> pd.DataFrame:
    if not api_key:
        api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        msg = "Odds API key 缺失"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()

    try:
        # 使用 The Odds API 的 v4 正式接口
        odds_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal"
        }
        odds_resp = requests.get(odds_url, params=params, timeout=30)
        
        if odds_resp.status_code == 401:
            msg = "Odds API 401 未授权，请检查 API Key 是否正确（注意：必须以字母开头，不要有多余空格）"
            if errors is not None:
                errors.append(msg)
            return pd.DataFrame()
            
        odds_resp.raise_for_status()
        odds_data = odds_resp.json()

        rows = []
        for game in odds_data:
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
        msg = f"Odds API HTTP 错误: {e}"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()
    except Exception as e:
        msg = f"Odds API 抓取异常: {e}"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()
