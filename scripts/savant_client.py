"""
Baseball Savant (Statcast) 客户端
通过 CSV 端点抓取指定时间段内的投球/击球数据
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

def fetch_savant_statcast(date_str: str = None) -> pd.DataFrame:
    """
    抓取 Baseball Savant 的 Statcast 数据。
    默认抓取最近7天数据，限制返回前1000行防过载。
    """
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
        "hfGT=R%7C&hfC=&hfSea=2025%7C&hfSit=&player_type=pitcher&hfOuts=&opponent=&"
        "pitcher_throws=&batter_stands=&hfSA=&game_date_gt={}&"
        "game_date_lt={}&hfFlag=&metric_1=&hfInn=&min_pitches=0&"
        "min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&type=details"
    ).format(start_str, end_str)

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        # 只保留关键列
        cols = ['pitch_type', 'release_speed', 'events', 'game_date']
        if all(c in df.columns for c in cols):
            return df[cols].head(1000)
        else:
            return df.head(1000)
    except Exception as e:
        print(f"Savant fetch error: {e}")
        return pd.DataFrame()
