"""
Odds-API.io 客户端（修正端点 + 自动清理 Key）
"""
import requests
import pandas as pd
import os

def fetch_odds(api_key: str = None, date_str: str = None, errors: list = None) -> pd.DataFrame:
    # 如果没有传入 api_key，从环境变量读取并自动清理换行符和空格
    if not api_key:
        raw = os.getenv("ODDS_API_KEY", "") or ""
        api_key = raw.strip().replace("\n", "").replace("\r", "")
    if not api_key:
        msg = "Odds API key missing"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()

    try:
        # 使用正确的 API 基础地址
        sports_url = "https://api.the-odds-api.com/v4/sports"
        headers = {"apikey": api_key}
        sports_resp = requests.get(sports_url, headers=headers, timeout=15)
        sports_resp.raise_for_status()
        sports_data = sports_resp.json()

        # 寻找 MLB 的 sport key
        baseball_key = None
        for sport in sports_data.get("data", sports_data if isinstance(sports_data, list) else []):
            if sport.get("group") == "Baseball" and "MLB" in sport.get("title", ""):
                baseball_key = sport["key"]
                break
        # 有些版本直接返回列表，所以兼容一下
        if not baseball_key and isinstance(sports_data, list):
            for sport in sports_data:
                if sport.get("group") == "Baseball" and "MLB" in sport.get("title", ""):
                    baseball_key = sport["key"]
                    break

        if not baseball_key:
            msg = "MLB key not found in Odds API"
            if errors is not None:
                errors.append(msg)
            return pd.DataFrame()

        # 获取赔率
        odds_url = f"https://api.the-odds-api.com/v4/sports/{baseball_key}/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "bookmakers": "bet365,draftkings"  # 你可以自行更换博彩公司
        }
        odds_resp = requests.get(odds_url, headers=headers, params=params, timeout=30)

        if odds_resp.status_code == 401:
            msg = "Odds API 401 未授权，请检查 API Key 是否正确（应以 oddsp- 开头，且无多余空格）"
            if errors is not None:
                errors.append(msg)
            return pd.DataFrame()

        odds_resp.raise_for_status()
        odds_data = odds_resp.json()

        if not odds_data.get("data") and not isinstance(odds_data, list):
            msg = "Odds API returned no game data (可能当天无 MLB 比赛，或博彩公司不支持)"
            if errors is not None:
                errors.append(msg)
            return pd.DataFrame()

        # 解析数据，兼容两种返回格式
        games = odds_data if isinstance(odds_data, list) else odds_data.get("data", [])
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
        msg = f"Odds API HTTP error: {e}"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()
    except Exception as e:
        msg = f"Odds API fetch error: {e}"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()
