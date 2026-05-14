"""
Baseball Savant (Statcast) 客户端
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

def fetch_savant_statcast(date_str: str = None, errors: list = None) -> pd.DataFrame:
    if not date_str:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=7)
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')
    else:
        start_str = date_str
        end_str = date_str

    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv?"
        "all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&"
        "hfGT=R%7C&hfC=&hfSea=2026%7C&hfSit=&player_type=pitcher&hfOuts=&opponent=&"
        "pitcher_throws=&batter_stands=&hfSA=&game_date_gt={}&"
        "game_date_lt={}&hfFlag=&metric_1=&hfInn=&min_pitches=0&"
        "min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&type=details"
    ).format(start_str, end_str)

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        cols = ['pitch_type', 'release_speed', 'events', 'game_date']
        if all(c in df.columns for c in cols):
            return df[cols].head(1000)
        else:
            return df.head(1000)
    except Exception as e:
        msg = f"Savant fetch error: {e}"
        if errors is not None:
            errors.append(msg)
        print(msg)
        return pd.DataFrame()
