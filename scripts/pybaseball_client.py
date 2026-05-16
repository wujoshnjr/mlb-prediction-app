"""
PyBaseball 客户端（延遲匯入，輕量化數據）
"""
import pandas as pd
from datetime import datetime, timedelta

def fetch_pybaseball(date_str: str = None, errors: list = None) -> dict:
    try:
        from pybaseball import statcast, batting_stats, pitching_stats
    except Exception as e:
        msg = f"PyBaseball import error: {e}"
        if errors is not None:
            errors.append(msg)
        return {'statcast_recent': pd.DataFrame(), 'batting_leaders': pd.DataFrame(), 'pitching_leaders': pd.DataFrame()}

    if not date_str:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
    else:
        start_str = date_str
        end_str = date_str

    try:
        # 只取少量數據避免記憶體不足
        sc = statcast(start_dt=start_str, end_dt=end_str)
        bat = batting_stats(2026)
        pitch = pitching_stats(2026)
        return {
            'statcast_recent': sc.head(100) if not sc.empty else pd.DataFrame(),
            'batting_leaders': bat.head(10) if not bat.empty else pd.DataFrame(),
            'pitching_leaders': pitch.head(10) if not pitch.empty else pd.DataFrame()
        }
    except Exception as e:
        msg = f"PyBaseball fetch error: {e}"
        if errors is not None:
            errors.append(msg)
        return {'statcast_recent': pd.DataFrame(), 'batting_leaders': pd.DataFrame(), 'pitching_leaders': pd.DataFrame()}
