import requests
import pandas as pd

def fetch_openmeteo(date_str: str = None, errors: list = None) -> pd.DataFrame:
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
        msg = f"Open-Meteo fetch error: {e}"
        if errors is not None:
            errors.append(msg)
        return pd.DataFrame()
