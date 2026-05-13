"""
Sportsipy 客户端
抓取球队战绩及示例球员数据
"""
from sportsipy.mlb.teams import Teams
from sportsipy.mlb.player import Player
import pandas as pd

def fetch_sportsipy(date_str: str = None) -> dict:
    """
    返回字典：
    - teams: 球队战绩 DataFrame
    - player_example: 指定球员的基本数据（这里以 Shohei Ohtani 为例）
    """
    try:
        teams_data = []
        for team in Teams(2025):
            teams_data.append({
                'name': team.name,
                'abbreviation': team.abbreviation,
                'wins': team.wins,
                'losses': team.losses,
                'win_percentage': team.win_percentage
            })
        df_teams = pd.DataFrame(teams_data)

        # 示例：大谷翔平
        try:
            player = Player('ohtansh01')
            player_info = {
                'name': player.name,
                'home_runs': player.home_runs,
                'era': getattr(player, 'era', None)  # 投手数据，打者可能没有
            }
        except:
            player_info = {'error': 'Player data unavailable'}

        return {
            'teams': df_teams,
            'player_example': player_info
        }
    except Exception as e:
        print(f"Sportsipy fetch error: {e}")
        return {'teams': pd.DataFrame(), 'player_example': {}}
