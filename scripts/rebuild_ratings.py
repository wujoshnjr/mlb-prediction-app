# scripts/rebuild_ratings.py
"""
一次性重建所有队伍的 ELO 和 Glicko2 评分。
从历史比赛数据中按日期顺序逐场更新，每场比赛只计算一次。
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
import config
from scripts.rating_updater import simple_elo_update, load_glicko2_league, save_glicko2_league, save_elo_ratings, load_elo_ratings
from scripts.glicko2_ratings import Glicko2League

# 强制使用 ELO 引擎进行重建（也可以选择 Glicko2）
ENGINE = 'elo'  # 或 'glicko2'

def rebuild():
    hist_file = "data/historical_predictions.csv"
    if not os.path.exists(hist_file):
        print("历史预测文件不存在")
        return

    df = pd.read_csv(hist_file)
    # 筛选已完成的比赛
    df = df[df['home_win'].notna()]
    if 'game_date' in df.columns:
        df = df.sort_values('game_date')
    else:
        df = df.sort_values('game_id')

    # 重置所有 rating
    if ENGINE == 'elo':
        ratings = {}
        for team in pd.concat([df['home_team'], df['away_team']]).unique():
            ratings[team] = 1500
        save_elo_ratings(ratings)
    else:
        league = Glicko2League()
        for team in pd.concat([df['home_team'], df['away_team']]).unique():
            league.add_team(team, rating=1500, rd=350, vol=0.06)
        save_glicko2_league(league)

    # 重建
    rated_ids = set()
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
        if ENGINE == 'elo':
            ratings = load_elo_ratings()
            simple_elo_update(ratings, home, away, home_score, away_score)
            save_elo_ratings(ratings)
        else:
            league = load_glicko2_league()
            # 类似逻辑...
            # 省略 Glicko2 重建细节，因为当前默认用 ELO

    print(f"重建完成，处理了 {len(rated_ids)} 场比赛")

if __name__ == '__main__':
    rebuild()
