"""
UnifiedSportsModel - 整合所有数据源（含错误收集）
"""
import os
import json
from datetime import datetime
import pandas as pd

# 导入所有数据抓取函数
from scripts.mlb_stats_client import fetch_mlb_statsapi
from scripts.savant_client import fetch_savant_statcast
from scripts.retro_client import fetch_retrosheet
from scripts.pybaseball_client import fetch_pybaseball
from scripts.sportsipy_client import fetch_sportsipy
from scripts.openmeteo_client import fetch_openmeteo
from scripts.balldontlie_client import fetch_balldontlie
from scripts.odds_client import fetch_odds


class UnifiedSportsModel:
    """
    整合 MLB Stats API、Baseball Savant、Retrosheet、
    PyBaseball、Sportsipy、Open-Meteo、Balldontlie、Odds-API.io
    """
    def __init__(self):
        self.ball_api_key = os.getenv("BALLDONTLIE_API_KEY")
        self.odds_api_key = os.getenv("ODDS_API_KEY")

    def gather_all_data(self, date_str: str = None) -> dict:
        """
        一次调用，拉取全部 8 个数据源，返回字典。
        同时收集每个数据源产生的错误信息（如果其函数支持 errors 参数）。
        """
        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')

        print(f"Starting data collection for {date_str}")

        # 用于收集错误信息的列表，会传给每个支持 errors 参数的函数
        errors_list = []

        # 调用数据源函数，若函数接受 errors 参数则传入
        # （若某个函数尚未更新为支持 errors，则忽略，不会报错）
        mlb_stats = self._call_fetch(fetch_mlb_statsapi, date_str, errors_list)
        savant = self._call_fetch(fetch_savant_statcast, date_str, errors_list)
        retro = self._call_fetch(fetch_retrosheet, date_str, errors_list)
        pyb = self._call_fetch(fetch_pybaseball, date_str, errors_list)
        sportsipy = self._call_fetch(fetch_sportsipy, date_str, errors_list)
        openmeteo = self._call_fetch(fetch_openmeteo, date_str, errors_list)
        balldontlie = self._call_fetch(fetch_balldontlie, self.ball_api_key, date_str, errors_list)
        odds = self._call_fetch(fetch_odds, self.odds_api_key, date_str, errors_list)

        # 构建返回结果（与之前完全一致）
        result = {
            'date': date_str,
            'mlb_statsapi': mlb_stats.to_dict(orient='records') if not mlb_stats.empty else [],
            'savant_statcast': savant.to_dict(orient='records') if not savant.empty else [],
            'retrosheet': retro.to_dict(orient='records') if not retro.empty else [],
            'pybaseball_statcast': pyb.get('statcast_recent', pd.DataFrame()).to_dict(orient='records') if not pyb.get('statcast_recent', pd.DataFrame()).empty else [],
            'pybaseball_batting': pyb.get('batting_leaders', pd.DataFrame()).to_dict(orient='records') if not pyb.get('batting_leaders', pd.DataFrame()).empty else [],
            'pybaseball_pitching': pyb.get('pitching_leaders', pd.DataFrame()).to_dict(orient='records') if not pyb.get('pitching_leaders', pd.DataFrame()).empty else [],
            'sportsipy_teams': sportsipy.get('teams', pd.DataFrame()).to_dict(orient='records') if not sportsipy.get('teams', pd.DataFrame()).empty else [],
            'sportsipy_player': sportsipy.get('player_example', {}),
            'openmeteo_weather': openmeteo.to_dict(orient='records') if not openmeteo.empty else [],
            'balldontlie_teams': balldontlie.to_dict(orient='records') if not balldontlie.empty else [],
            'odds_data': odds.to_dict(orient='records') if not odds.empty else [],
            'errors': errors_list  # 收集到的所有错误信息
        }

        # 保存报告至 report 目录
        if os.path.isfile('report'):
            os.remove('report')
        os.makedirs('report', exist_ok=True)
        with open(f'report/{date_str}.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)

        return result

    def _call_fetch(self, func, *args):
        """
        调用数据抓取函数，若函数签名支持 errors 关键字参数则传入 errors_list。
        这样即使某个函数尚未更新，也不会报错。
        """
        import inspect
        try:
            sig = inspect.signature(func)
            if 'errors' in sig.parameters:
                return func(*args, errors=errors_list)  # 注意这里传入 errors_list（之前定义的）
            else:
                return func(*args)
        except Exception:
            return func(*args)
