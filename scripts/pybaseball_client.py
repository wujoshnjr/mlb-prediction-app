"""
PyBaseball 客户端
整合 Statcast 近期数据、赛季打击/投手统计
"""
import pandas as pd
from pybaseball import statcast, batting_stats, pitching_stats
from datetime import datetime, timedelta

def fetch_pybaseball(date_str: str = None) -> dict:
    """
    返回一个字典，包含：
    - statcast_recent: 近7天 Statcast 数据
    - batting_leaders: 打击排行榜
    - pitching_leaders: 投手排行榜
    """
    if not date_str:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
    else:
        start_str = date_str
        end_str = date_str

    try:
        sc = statcast(start_dt=start_str, end_dt=end_str)
        bat = batting_stats(2025)
        pitch = pitching_stats(2025)

        return {
            'statcast_recent': sc.head(500) if not sc.empty else pd.DataFrame(),
            'batting_leaders': bat.head(10) if not bat.empty else pd.DataFrame(),
            'pitching_leaders': pitch.head(10) if not pitch.empty else pd.DataFrame()
        }
    except Exception as e:
        print(f"PyBaseball fetch error: {e}")
        return {
            'statcast_recent': pd.DataFrame(),
            'batting_leaders': pd.DataFrame(),
            'pitching_leaders': pd.DataFrame()
        }
