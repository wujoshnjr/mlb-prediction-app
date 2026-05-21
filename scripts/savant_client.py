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
        "https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones="
        f"&hfGT=R%7C&hfC=&hfSea=2026%7C&hfSit=&player_type=pitcher&hfOuts=&opponent="
        f"&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={start_str}&game_date_lt={end_str}"
        "&hfFlag=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name&sort_col=pitches"
        "&player_event_sort=h_launch_speed&sort_order=desc&type=details"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        desired = [
            'pitch_type', 'release_speed', 'events', 'game_date',
            'launch_speed', 'launch_angle', 'barrel', 'hard_hit',
            'hit_distance_sc', 'expected_batting_avg', 'expected_slugging_percent',
            'expected_woba', 'pitch_hand', 'bat_side',
            'pfx_x', 'pfx_z',
            'release_spin_rate',
            'plate_x', 'plate_z',
            'bat_speed'            # 新增挥棒速度
        ]
        cols = [c for c in desired if c in df.columns]
        return df[cols].head(3000)
    except Exception as e:
        if errors is not None:
            errors.append(f"Savant fetch error: {e}")
        return pd.DataFrame()
