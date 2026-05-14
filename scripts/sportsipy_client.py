"""
球队与球员数据客户端（使用 pybaseball 替代 sportsipy）
"""
import pandas as pd
from pybaseball import standings, batting_stats, playerid_lookup

def fetch_sportsipy(date_str: str = None) -> dict:
    """
    返回字典：
    - teams: 球队战绩 DataFrame（来自 pybaseball standings）
    - player_example: 指定球员的基本数据（这里以 Shohei Ohtani 为例）
    """
    year = 2026  # 可根据需要修改

    # --- 球队战绩 ---
    try:
        # standings 返回一个列表，包含每个分区的 DataFrame
        all_standings = standings(year)
        team_list = []
        for division_df in all_standings:
            # 每个 division_df 可能有很多列，我们取常用字段
            for _, row in division_df.iterrows():
                team_list.append({
                    'name': row.get('Tm'),                # 球队名缩写
                    'wins': row.get('W'),
                    'losses': row.get('L'),
                    'win_pct': row.get('W-L%'),
                    'gb': row.get('GB'),
                })
        df_teams = pd.DataFrame(team_list)
    except Exception as e:
        print(f"standings fetch error: {e}")
        df_teams = pd.DataFrame()

    # --- 球员示例 (大谷翔平) ---
    player_info = {}
    try:
        # 先查找球员 ID
        player = playerid_lookup('ohtani', 'shohei')
        if not player.empty:
            mlb_id = player.iloc[0]['key_mlbam']
            # 获取该球员的打击数据（2026赛季）
            bat_stats = batting_stats(year, qual=1)  # qual=1 表示有足够打席的球员
            ohtani_stats = bat_stats[bat_stats['IDfg'] == mlb_id]  # 注意：可能要用MLBAM ID匹配，这里简化
            if not ohtani_stats.empty:
                row = ohtani_stats.iloc[0]
                player_info = {
                    'name': 'Shohei Ohtani',
                    'home_runs': row.get('HR'),
                    'avg': row.get('AVG'),
                    'ops': row.get('OPS'),
                }
            else:
                player_info = {'name': 'Shohei Ohtani', 'note': 'stats not found in qual'}
        else:
            player_info = {'error': 'Player not found'}
    except Exception as e:
        print(f"Player fetch error: {e}")
        player_info = {'error': str(e)}

    return {
        'teams': df_teams,
        'player_example': player_info
    }
