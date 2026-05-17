"""
UnifiedSportsModel - 整合所有数据源（启动友善、自动清理 API Key）
"""
import os
import json
from datetime import datetime
import pandas as pd

# 防禦性匯入：任何一個模組失敗都不影響主服務啟動
fetch_mlb_statsapi = None
fetch_savant_statcast = None
fetch_retrosheet = None
fetch_pybaseball = None
fetch_sportsipy = None
fetch_openmeteo = None
fetch_balldontlie = None
fetch_odds = None

try:
    from scripts.mlb_stats_client import fetch_mlb_statsapi
except Exception as e:
    print(f"Warning: Failed to import mlb_stats_client: {e}")

try:
    from scripts.savant_client import fetch_savant_statcast
except Exception as e:
    print(f"Warning: Failed to import savant_client: {e}")

try:
    from scripts.retro_client import fetch_retrosheet
except Exception as e:
    print(f"Warning: Failed to import retro_client: {e}")

try:
    from scripts.pybaseball_client import fetch_pybaseball
except Exception as e:
    print(f"Warning: Failed to import pybaseball_client: {e}")

try:
    from scripts.sportsipy_client import fetch_sportsipy
except Exception as e:
    print(f"Warning: Failed to import sportsipy_client: {e}")

try:
    from scripts.openmeteo_client import fetch_openmeteo
except Exception as e:
    print(f"Warning: Failed to import openmeteo_client: {e}")

try:
    from scripts.balldontlie_client import fetch_balldontlie
except Exception as e:
    print(f"Warning: Failed to import balldontlie_client: {e}")

try:
    from scripts.odds_client import fetch_odds
except Exception as e:
    print(f"Warning: Failed to import odds_client: {e}")


class UnifiedSportsModel:
    def __init__(self):
        # 自動清理 Key 中的換行與空白
        raw_ball = os.getenv("BALLDONTLIE_API_KEY", "") or ""
        raw_odds = os.getenv("ODDS_API_KEY", "") or ""
        self.ball_api_key = raw_ball.strip().replace("\n", "").replace("\r", "")
        self.odds_api_key = raw_odds.strip().replace("\n", "").replace("\r", "")

    def gather_all_data(self, date_str: str = None) -> dict:
        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')

        errors = []
        result = {
            'date': date_str,
            'mlb_statsapi': [], 'savant_statcast': [], 'retrosheet': [],
            'pybaseball_statcast': [], 'pybaseball_batting': [], 'pybaseball_pitching': [],
            'sportsipy_teams': [], 'sportsipy_player': {},
            'openmeteo_weather': [], 'balldontlie_teams': [], 'odds_data': [],
            'errors': errors
        }

        def safe_call(func, name, *args):
            if func is None:
                errors.append(f"{name} module not loaded.")
                return pd.DataFrame() if name not in ("pybaseball", "sportsipy") else {}
            try:
                res = func(*args)
                return res
            except Exception as e:
                errors.append(f"{name} fetch error: {e}")
                return pd.DataFrame() if name not in ("pybaseball", "sportsipy") else {}

        mlb_stats = safe_call(fetch_mlb_statsapi, "mlb_statsapi", date_str, errors)
        savant = safe_call(fetch_savant_statcast, "savant_statcast", date_str, errors)
        retro = safe_call(fetch_retrosheet, "retrosheet", date_str, errors)
        pyb = safe_call(fetch_pybaseball, "pybaseball", date_str, errors)
        sportsipy = safe_call(fetch_sportsipy, "sportsipy", date_str, errors)
        openmeteo = safe_call(fetch_openmeteo, "openmeteo", date_str, errors)
        balldontlie = safe_call(fetch_balldontlie, "balldontlie", self.ball_api_key, date_str, errors)
        odds = safe_call(fetch_odds, "odds", self.odds_api_key, date_str, errors)

        # 填充結果
        result['mlb_statsapi'] = mlb_stats.to_dict(orient='records') if not mlb_stats.empty else []
        result['savant_statcast'] = savant.to_dict(orient='records') if not savant.empty else []
        result['retrosheet'] = retro.to_dict(orient='records') if not retro.empty else []

        if isinstance(pyb, dict):
            result['pybaseball_statcast'] = pyb.get('statcast_recent', pd.DataFrame()).to_dict(orient='records') if not pyb.get('statcast_recent', pd.DataFrame()).empty else []
            result['pybaseball_batting'] = pyb.get('batting_leaders', pd.DataFrame()).to_dict(orient='records') if not pyb.get('batting_leaders', pd.DataFrame()).empty else []
            result['pybaseball_pitching'] = pyb.get('pitching_leaders', pd.DataFrame()).to_dict(orient='records') if not pyb.get('pitching_leaders', pd.DataFrame()).empty else []

        if isinstance(sportsipy, dict):
            result['sportsipy_teams'] = sportsipy.get('teams', pd.DataFrame()).to_dict(orient='records') if not sportsipy.get('teams', pd.DataFrame()).empty else []
            result['sportsipy_player'] = sportsipy.get('player_example', {})

        result['openmeteo_weather'] = openmeteo.to_dict(orient='records') if not openmeteo.empty else []
        result['balldontlie_teams'] = balldontlie.to_dict(orient='records') if not balldontlie.empty else []
        result['odds_data'] = odds.to_dict(orient='records') if not odds.empty else []

        # 保存報告
        if os.path.isfile('report'):
            os.remove('report')
        os.makedirs('report', exist_ok=True)
        with open(f'report/{date_str}.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)

        return result
