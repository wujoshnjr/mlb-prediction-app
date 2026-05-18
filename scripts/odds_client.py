"""
Odds-API.io 客户端 (v3)
使用官方推荐的 Python SDK 来获取 MLB 的胜平负赔率
"""
import os
import pandas as pd
from odds_api import OddsAPIClient

def fetch_odds(api_key: str = None, date_str: str = None, errors: list = None) -> pd.DataFrame:
    """
    使用 SDK 从 Odds-API.io 获取 MLB 的胜平负赔率。
    返回一个 Pandas DataFrame。
    """
    # 1. 安全地拿到 API 密钥
    if not api_key:
        api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        if errors is not None:
            errors.append("Odds API 密钥未找到。")
        return pd.DataFrame()

    try:
        # 2. 建立 API 客户端
        client = OddsAPIClient(api_key=api_key)

        # 3. 获取 MLB 的赛事列表
        events = client.get_events(sport="baseball", league="usa-mlb")
        if not events:
            if errors is not None:
                errors.append("没有找到 MLB 赛事。")
            client.close()
            return pd.DataFrame()

        # 4. 收集赛事 ID
        event_ids = [event.get("id") for event in events if event.get("id")]
        if not event_ids:
            client.close()
            return pd.DataFrame()

        # 5. 获取这些赛事的赔率
        odds_response = client.get_odds(
            event_ids=event_ids,
            bookmakers="bet365,draftkings",
            markets="ML"
        )

        # 6. 解析数据
        all_rows = []
        if isinstance(odds_response, list):
            for event_odds in odds_response:
                home_team = event_odds.get("home", "Unknown")
                away_team = event_odds.get("away", "Unknown")
                bookmakers = event_odds.get("bookmakers", {})
                
                for bookmaker_name, markets in bookmakers.items():
                    for market in markets:
                        if market.get("name") == "ML":
                            for odd_data in market.get("odds", []):
                                all_rows.append({
                                    "home_team": home_team,
                                    "away_team": away_team,
                                    "bookmaker": bookmaker_name,
                                    "bet_type": "ML",
                                    "team": "home", "odds": odd_data.get("home")
                                })
                                all_rows.append({
                                    "home_team": home_team,
                                    "away_team": away_team,
                                    "bookmaker": bookmaker_name,
                                    "bet_type": "ML",
                                    "team": "away", "odds": odd_data.get("away")
                                })

        client.close()
        return pd.DataFrame(all_rows)

    except Exception as e:
        if errors is not None:
            errors.append(f"获取赔率时发生错误: {e}")
        return pd.DataFrame()
