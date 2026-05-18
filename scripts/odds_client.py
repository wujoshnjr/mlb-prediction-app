"""
Odds-API.io 客户端 (v3 - 直接 requests 调用)
使用官方 v3 API 获取 MLB 的胜平负赔率
"""
import requests
import pandas as pd
import os

def fetch_odds(api_key: str = None, date_str: str = None, errors: list = None) -> pd.DataFrame:
    if not api_key:
        api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        if errors is not None:
            errors.append("Odds API key missing")
        return pd.DataFrame()

    try:
        # 第一步：获取 MLB 的赛事列表
        events_url = "https://api.odds-api.io/v3/events"
        params = {
            "apiKey": api_key,
            "sport": "baseball",       # 根据你刚才获取的列表，baseball 是正确的
            "league": "usa-mlb"
        }
        resp = requests.get(events_url, params=params, timeout=15)
        if resp.status_code == 401:
            if errors is not None:
                errors.append("Odds API 401: Invalid API key")
            return pd.DataFrame()
        resp.raise_for_status()
        events_data = resp.json()
        events = events_data if isinstance(events_data, list) else events_data.get("data", [])

        if not events:
            if errors is not None:
                errors.append("No MLB events found in Odds API")
            return pd.DataFrame()

        event_ids = [e["id"] for e in events if "id" in e]
        if not event_ids:
            if errors is not None:
                errors.append("No event IDs found")
            return pd.DataFrame()

        # 第二步：获取这些赛事的赔率
        odds_url = "https://api.odds-api.io/v3/odds"
        odds_params = {
            "apiKey": api_key,
            "event_ids": ",".join(event_ids),   # 多个ID用逗号连接
            "bookmakers": "bet365,draftkings",
            "markets": "ML"                     # 胜平负市场
        }
        odds_resp = requests.get(odds_url, params=odds_params, timeout=30)
        odds_resp.raise_for_status()
        odds_data = odds_resp.json()

        # 第三步：解析为 DataFrame
        rows = []
        # API 可能直接返回列表或包含 data 的字典
        games = odds_data if isinstance(odds_data, list) else odds_data.get("data", [])
        for game in games:
            home = game.get("home", "Unknown")
            away = game.get("away", "Unknown")
            bookmakers = game.get("bookmakers", {})
            for book_name, markets in bookmakers.items():
                for market in markets:
                    if market.get("name") == "ML":
                        for odd in market.get("odds", []):
                            rows.append({
                                "home_team": home,
                                "away_team": away,
                                "bookmaker": book_name,
                                "bet_type": "ML",
                                "team": "home",
                                "odds": odd.get("home")
                            })
                            rows.append({
                                "home_team": home,
                                "away_team": away,
                                "bookmaker": book_name,
                                "bet_type": "ML",
                                "team": "away",
                                "odds": odd.get("away")
                            })
        return pd.DataFrame(rows)

    except Exception as e:
        if errors is not None:
            errors.append(f"获取赔率时发生错误: {e}")
        return pd.DataFrame()
