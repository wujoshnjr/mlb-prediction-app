"""
UnifiedSportsModel - 整合所有数据源
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
        一次调用，拉取全部8个数据源，返回字典
        """
        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')

        print(f"Starting data collection for {date_str}")

        # 1. MLB Stats API
        mlb_stats = fetch_mlb_statsapi(date_str)
        # 2. Baseball Savant
        savant = fetch_savant_statcast(date_str)
        # 3. Retrosheet
        retro = fetch_retrosheet(date_str)
        # 4. PyBaseball
        pyb = fetch_pybaseball(date_str)
        # 5. Sportsipy
        sportsipy = fetch_sportsipy(date_str)
        # 6. Open-Meteo
        openmeteo = fetch_openmeteo(date_str)
        # 7. Balldontlie
        balldontlie = fetch_balldontlie(self.ball_api_key, date_str)
        # 8. Odds-API.io
        odds = fetch_odds(self.odds_api_key, date_str)

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
            'odds_data': odds.to_dict(orient='records') if not odds.empty else []
        }

        # 可选：保存至 report 目录
        os.makedirs('report', exist_ok=True)
        with open(f'report/{date_str}.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)

        return result
