"""
PyBaseball 客户端（延迟导入，防止启动崩溃）
"""
import pandas as pd
from datetime import datetime, timedelta

def fetch_pybaseball(date_str: str = None, errors: list = None) -> dict:
    try:
        from pybaseball import statcast, batting_stats, pitching_stats
        # 设置伪装头，避免 403
        from pybaseball import cache
        cache._HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
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
        sc = statcast(start_dt=start_str, end_dt=end_str)
        bat = batting_stats(2026)
        pitch = pitching_stats(2026)
        return {
            'statcast_recent': sc.head(500) if not sc.empty else pd.DataFrame(),
            'batting_leaders': bat.head(10) if not bat.empty else pd.DataFrame(),
            'pitching_leaders': pitch.head(10) if not pitch.empty else pd.DataFrame()
        }
    except Exception as e:
        msg = f"PyBaseball fetch error: {e}"
        if errors is not None:
            errors.append(msg)
        return {'statcast_recent': pd.DataFrame(), 'batting_leaders': pd.DataFrame(), 'pitching_leaders': pd.DataFrame()}
