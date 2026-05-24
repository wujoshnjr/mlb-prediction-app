# scripts/rebuild_ratings.py
"""
一次性重建所有队伍的 ELO 和 Glicko2 评分。
从历史预测数据中按日期顺序逐场更新，每场只计算一次。
使用完整统一队名，输出健康报告。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import config
from scripts.rating_updater import simple_elo_update, load_glicko2_league, save_glicko2_league, save_elo_ratings, load_elo_ratings
from scripts.glicko2_ratings import Glicko2League

# 統一名稱映射（與 prediction.py 一致）
TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "D-backs", "Atlanta Braves": "Braves",
    "Baltimore Orioles": "Orioles", "Boston Red Sox": "Red Sox",
    "Chicago Cubs": "Cubs", "Chicago White Sox": "White Sox",
    "Cincinnati Reds": "Reds", "Cleveland Guardians": "Guardians",
    "Colorado Rockies": "Rockies", "Detroit Tigers": "Tigers",
    "Houston Astros": "Astros", "Kansas City Royals": "Royals",
    "Los Angeles Angels": "Angels", "Los Angeles Dodgers": "Dodgers",
    "Miami Marlins": "Marlins", "Milwaukee Brewers": "Brewers",
    "Minnesota Twins": "Twins", "New York Mets": "Mets",
    "New York Yankees": "Yankees", "Oakland Athletics": "Athletics",
    "Philadelphia Phillies": "Phillies", "Pittsburgh Pirates": "Pirates",
    "San Diego Padres": "Padres", "San Francisco Giants": "Giants",
    "Seattle Mariners": "Mariners", "St. Louis Cardinals": "Cardinals",
    "Tampa Bay Rays": "Rays", "Texas Rangers": "Rangers",
    "Toronto Blue Jays": "Blue Jays", "Washington Nationals": "Nationals",
    "D-backs": "D-backs", "Braves": "Braves", "Orioles": "Orioles",
    "Red Sox": "Red Sox", "Cubs": "Cubs", "White Sox": "White Sox",
    "Reds": "Reds", "Guardians": "Guardians", "Rockies": "Rockies",
    "Tigers": "Tigers", "Astros": "Astros", "Royals": "Royals",
    "Angels": "Angels", "Dodgers": "Dodgers", "Marlins": "Marlins",
    "Brewers": "Brewers", "Twins": "Twins", "Mets": "Mets",
    "Yankees": "Yankees", "Athletics": "Athletics", "Phillies": "Phillies",
    "Pirates": "Pirates", "Padres": "Padres", "Giants": "Giants",
    "Mariners": "Mariners", "Cardinals": "Cardinals", "Rays": "Rays",
    "Rangers": "Rangers", "Blue Jays": "Blue Jays", "Nationals": "Nationals"
}

def rebuild(engine='elo'):
    hist_file = "data/historical_predictions.csv"
    if not os.path.exists(hist_file):
        print("历史预测文件不存在")
        return

    df = pd.read_csv(hist_file)
    df = df[df['home_win'].notna()]  # 只取已完赛
    if 'game_date' in df.columns:
        df = df.sort_values('game_date')
    else:
        df = df.sort_values('game_id')

    # 统一队名
    df['home_team'] = df['home_team'].map(TEAM_NAME_MAP).fillna(df['home_team'])
    df['away_team'] = df['away_team'].map(TEAM_NAME_MAP).fillna(df['away_team'])

    # 初始化
    if engine == 'elo':
        ratings = {team: 1500 for team in pd.concat([df['home_team'], df['away_team']]).unique()}
        save_elo_ratings(ratings)
    else:
        league = Glicko2League()
        for team in pd.concat([df['home_team'], df['away_team']]).unique():
            league.add_team(team, rating=1500, rd=350, vol=0.06)
        save_glicko2_league(league)

    rated_ids = set()
    processed = 0
    for _, row in df.iterrows():
        gid = str(row.get('game_id'))
        if gid in rated_ids:
            continue
        rated_ids.add(gid)
        home = row['home_team']
        away = row['away_team']
        home_score = int(row['home_score']) if 'home_score' in row and not pd.isna(row['home_score']) else None
        away_score = int(row['away_score']) if 'away_score' in row and not pd.isna(row['away_score']) else None
        if home_score is None or away_score is None:
            continue
        if engine == 'elo':
            ratings = load_elo_ratings()
            simple_elo_update(ratings, home, away, home_score, away_score)
            save_elo_ratings(ratings)
        else:
            league = load_glicko2_league()
            home_team = league.teams[home]
            away_team = league.teams[away]
            # 使用深拷贝避免顺序污染
            import copy
            home_opponent = copy.deepcopy(away_team)
            away_opponent = copy.deepcopy(home_team)
            if home_score > away_score:
                home_team.update(home_opponent, 1.0)
                away_team.update(away_opponent, 0.0)
            elif home_score < away_score:
                home_team.update(home_opponent, 0.0)
                away_team.update(away_opponent, 1.0)
            else:
                home_team.update(home_opponent, 0.5)
                away_team.update(away_opponent, 0.5)
            save_glicko2_league(league)
        processed += 1

    # 健康报告
    if engine == 'elo':
        ratings = load_elo_ratings()
        values = list(ratings.values())
        print(f"Elo 重建完成，处理比赛数: {processed}")
        print(f"Rating min: {min(values):.1f}, max: {max(values):.1f}, range: {max(values)-min(values):.1f}")
        # 检查重名
        print(f"球队数量: {len(ratings)}")
    else:
        league = load_glicko2_league()
        values = [t.rating for t in league.teams.values()]
        print(f"Glicko2 重建完成，处理比赛数: {processed}")
        print(f"Rating min: {min(values):.1f}, max: {max(values):.1f}, range: {max(values)-min(values):.1f}")

    # 保存 rated_game_ids
    with open('data/rated_game_ids.json', 'w') as f:
        json.dump(list(rated_ids), f)
    print(f"已保存 {len(rated_ids)} 个已处理 game_id")

if __name__ == '__main__':
    rebuild('elo')   # 或 'glicko2'
