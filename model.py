"""
UnifiedSportsModel - 整合所有数据源（启动友好，坚韧版）
"""
import os
import json
from datetime import datetime
import pandas as pd

# 定义所有需要导入的模块和函数，默认设为 None
fetch_mlb_statsapi = None
fetch_savant_statcast = None
fetch_retrosheet = None
fetch_pybaseball = None
fetch_sportsipy = None
fetch_openmeteo = None
fetch_balldontlie = None
fetch_odds = None

# 依次尝试导入，任何一个失败都不会影响主服务启动
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
        self.ball_api_key = os.getenv("BALLDONTLIE_API_KEY")
        self.odds_api_key = os.getenv("ODDS_API_KEY")

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

        # 用一个通用的安全调用函数来执行抓取
        def safe_call(func, name, *args):
            if func is None:
                errors.append(f"{name} module not loaded.")
                return pd.DataFrame()
            try:
                res = func(*args)
                return res
            except Exception as e:
                errors.append(f"{name} fetch error: {e}")
                return pd.DataFrame()

        # 安全调用所有数据源
        mlb_stats = safe_call(fetch_mlb_statsapi, "mlb_statsapi", date_str, errors)
        savant = safe_call(fetch_savant_statcast, "savant_statcast", date_str, errors)
        retro = safe_call(fetch_retrosheet, "retrosheet", date_str, errors)
        pyb = safe_call(fetch_pybaseball, "pybaseball", date_str, errors)
        sportsipy = safe_call(fetch_sportsipy, "sportsipy", date_str, errors)
        openmeteo = safe_call(fetch_openmeteo, "openmeteo", date_str, errors)
        balldontlie = safe_call(fetch_balldontlie, "balldontlie", self.ball_api_key, date_str, errors)
        odds = safe_call(fetch_odds, "odds", self.odds_api_key, date_str, errors)

        # ... 从这里开始，把上面获取到的变量放入 result 字典里，这部分逻辑和你之前的完全一样 ...
        # ... 为了简洁，我在这里省略，你可以把你原有代码中构建 result 的部分复制过来 ...
        
        # 保存报告
        if os.path.isfile('report'):
            os.remove('report')
        os.makedirs('report', exist_ok=True)
        with open(f'report/{date_str}.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)

        return result
