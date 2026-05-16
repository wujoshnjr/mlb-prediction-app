"""
球隊與球員數據客戶端（使用 pybaseball，強制請求頭）
"""
import pandas as pd

def fetch_sportsipy(date_str: str = None, errors: list = None) -> dict:
    try:
        from pybaseball import standings, batting_stats, playerid_lookup, cache
        # 強制設定偽裝頭（與 pybaseball 一致）
        cache._HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.fangraphs.com/',
        }
    except Exception as e:
        msg = f"Sportsipy import error: {e}"
        if errors is not None:
            errors.append(msg)
        return {'teams': pd.DataFrame(), 'player_example': {}}

    year = 2026

    # 球队战绩
    try:
        all_standings = standings(year)
        team_list = []
        for division_df in all_standings:
            for _, row in division_df.iterrows():
                team_list.append({
                    'name': row.get('Tm'),
                    'wins': row.get('W'),
                    'losses': row.get('L'),
                    'win_pct': row.get('W-L%'),
                    'gb': row.get('GB'),
                })
        df_teams = pd.DataFrame(team_list)
    except Exception as e:
        msg = f"Sportsipy teams error: {e}"
        if errors is not None:
            errors.append(msg)
        df_teams = pd.DataFrame()

    # 球员示例
    player_info = {}
    try:
        player = playerid_lookup('ohtani', 'shohei')
        if not player.empty:
            mlb_id = player.iloc[0]['key_mlbam']
            bat_stats = batting_stats(year, qual=1)
            ohtani_stats = bat_stats[bat_stats['IDfg'] == mlb_id]
            if not ohtani_stats.empty:
                row = ohtani_stats.iloc[0]
                player_info = {
                    'name': 'Shohei Ohtani',
                    'home_runs': row.get('HR'),
                    'avg': row.get('AVG'),
                    'ops': row.get('OPS'),
                }
            else:
                player_info = {'name': 'Shohei Ohtani', 'note': 'stats not found'}
        else:
            player_info = {'error': 'Player not found'}
    except Exception as e:
        msg = f"Sportsipy player error: {e}"
        if errors is not None:
            errors.append(msg)
        player_info = {'error': str(e)}

    return {'teams': df_teams, 'player_example': player_info}
