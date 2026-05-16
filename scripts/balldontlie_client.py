"""
BALLDONTLIE 客户端
"""
import requests
import pandas as pd
import os

def fetch_balldontlie(api_key: str = None, date_str: str = None, errors: list = None) -> pd.DataFrame:
    if not api_key:
        api_key = os.getenv("BALLDONTLIE_API_KEY")
    if not api_key:
        msg = "Balldontlie API key missing"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()

    headers = {"Authorization": api_key}
    try:
        resp = requests.get("https://api.balldontlie.io/mlb/v1/teams", headers=headers, timeout=15)
        resp.raise_for_status()
        teams = resp.json().get('data', [])
        return pd.DataFrame(teams)[['id', 'name', 'division', 'league']]
    except Exception as e:
        msg = f"Balldontlie fetch error: {e}"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()
