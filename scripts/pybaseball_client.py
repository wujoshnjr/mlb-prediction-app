"""
PyBaseball 客户端（強制請求頭修復 403）
"""
import pandas as pd
from datetime import datetime, timedelta

def fetch_pybaseball(date_str: str = None, errors: list = None) -> dict:
    try:
        from pybaseball import statcast, batting_stats, pitching_stats, cache
        # 強制設定所有請求的偽裝頭（修復 Fangraphs 403）
        cache._HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.fangraphs.com/',
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
            'statcast_recent': sc.head(100) if not sc.empty else pd.DataFrame(),
            'batting_leaders': bat.head(10) if not bat.empty else pd.DataFrame(),
            'pitching_leaders': pitch.head(10) if not pitch.empty else pd.DataFrame()
        }
    except Exception as e:
        msg = f"PyBaseball fetch error: {e}"
        if errors is not None:
            errors.append(msg)
        return {'statcast_recent': pd.DataFrame(), 'batting_leaders': pd.DataFrame(), 'pitching_leaders': pd.DataFrame()}
