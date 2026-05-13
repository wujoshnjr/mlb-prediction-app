"""
Odds-API.io 客户端
需要 API Key，从环境变量 ODDS_API_KEY 读取
"""
import pandas as pd
import os
from odds_api_io import OddsAPIClient

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
        client = OddsAPIClient(api_key=api_key)
        # 先获取体育列表，找到 MLB 的 key
        sports = client.sports_list()
        baseball_key = None
        for sport in sports.get('data', []):
            if sport.get('group') == 'Baseball' and 'MLB' in sport.get('title', ''):
                baseball_key = sport['key']
                break

        if not baseball_key:
            print("MLB key not found in Odds API")
            return pd.DataFrame()

        odds_data = client.odds(sport=baseball_key, regions='us')
        return pd.DataFrame(odds_data.get('data', []))
    except Exception as e:
        print(f"Odds API fetch error: {e}")
        return pd.DataFrame()
