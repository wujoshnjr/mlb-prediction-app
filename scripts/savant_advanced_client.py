import requests
import pandas as pd
from io import StringIO

def fetch_savant_advanced(start_date: str, end_date: str, errors: list = None) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv?"
        "all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&"
        "hfGT=R%7C&hfC=&hfSea=&hfSit=&player_type=pitcher&hfOuts=&opponent=&"
        "pitcher_throws=&batter_stands=&hfSA=&game_date_gt={start}&"
        "game_date_lt={end}&hfFlag=&metric_1=&hfInn=&min_pitches=0&"
        "min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&type=details"
    ).format(start=start_date, end=end_date)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return pd.read_csv(StringIO(resp.text), low_memory=False)
    except Exception as e:
        if errors is not None:
            errors.append(f"Advanced Savant fetch error: {e}")
        return pd.DataFrame()
