"""
MLB Stats API 客户端
使用 mlb-statsapi 包获取实时赛程与比分
"""
import pandas as pd
from mlb_statsapi import statsapi

def fetch_mlb_statsapi(date_str: str = None) -> pd.DataFrame:
    """
    获取指定日期的赛程，若无日期则获取当日赛程
    返回包含 game_id, home, away, status 的 DataFrame
    """
    params = {}
    if date_str:
        params['date'] = date_str

    schedule = statsapi.get('schedule', params)
    games = []

    for date_info in schedule.get('dates', []):
        for game in date_info.get('games', []):
            games.append({
                'game_id': game.get('gamePk'),
                'home': game['teams']['home']['team']['name'],
                'away': game['teams']['away']['team']['name'],
                'status': game.get('status', {}).get('abstractGameState')
            })

    return pd.DataFrame(games)
