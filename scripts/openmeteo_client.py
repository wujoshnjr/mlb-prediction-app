"""
Open-Meteo 客户端
获取指定坐标（洋基球场）的天气预报
"""
import requests
import pandas as pd

def fetch_openmeteo(date_str: str = None) -> pd.DataFrame:
    """
    返回每小时温度与降水量 DataFrame
    """
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=40.8296&longitude=-73.9262"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=1&timezone=America/New_York"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get('hourly', {})
        return pd.DataFrame({
            'time': hourly.get('time', []),
            'temperature_2m': hourly.get('temperature_2m', []),
            'precipitation': hourly.get('precipitation', [])
        })
    except Exception as e:
        print(f"Open-Meteo fetch error: {e}")
        return pd.DataFrame()
