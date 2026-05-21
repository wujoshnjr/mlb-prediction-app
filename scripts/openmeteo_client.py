import requests
import pandas as pd

def fetch_openmeteo(date_str: str = None, errors: list = None) -> pd.DataFrame:
    url = ("https://api.open-meteo.com/v1/forecast?latitude=40.8296&longitude=-73.9262"
           "&hourly=temperature_2m,precipitation,wind_speed_10m,wind_direction_10m"
           "&forecast_days=1&timezone=America/New_York")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        hourly = resp.json().get("hourly",{})
        return pd.DataFrame({
            "time": hourly.get("time",[]),
            "temperature_2m": hourly.get("temperature_2m",[]),
            "precipitation": hourly.get("precipitation",[]),
            "wind_speed": hourly.get("wind_speed_10m",[]),
            "wind_direction": hourly.get("wind_direction_10m",[])
        })
    except Exception as e:
        if errors is not None:
            errors.append(f"Open-Meteo fetch error: {e}")
        return pd.DataFrame()
