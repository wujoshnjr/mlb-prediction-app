"""
Balldontlie 客户端
需要 API Key，从环境变量 BALLDONTLIE_API_KEY 读取
"""
import requests
import pandas as pd
import os

def fetch_balldontlie(api_key: str = None, date_str: str = None) -> pd.DataFrame:
    """
    从 Balldontlie 获取 MLB 球队信息
    """
    if not api_key:
        api_key = os.getenv("BALLDONTLIE_API_KEY")
    if not api_key:
        print("Balldontlie API key missing")
        return pd.DataFrame()

    headers = {"Authorization": api_key}
    try:
        resp = requests.get("https://api.balldontlie.io/mlb/v1/teams", headers=headers, timeout=15)
        resp.raise_for_status()
        teams = resp.json().get('data', [])
        return pd.DataFrame(teams)[['id', 'name', 'division', 'league']]
    except Exception as e:
        print(f"Balldontlie fetch error: {e}")
        return pd.DataFrame()
