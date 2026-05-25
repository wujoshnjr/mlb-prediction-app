# scripts/rebuild_ratings.py
"""
一次性重建所有队伍的 ELO 与 Glicko2 评分。
从历史预测 CSV 与 data/historical/*.parquet 收集已有比分的 finalized games。
只处理 home_score/away_score 均存在的比赛。
输出干净评分文件及 rated_game_ids.json。
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import config
from scripts.rating_updater import simple_elo_update, load_glicko2_league, save_glicko2_league, save_elo_ratings, load_elo_ratings
from scripts.glicko2_ratings import Glicko2League

# 统一的队名映射
TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "D-backs", "Diamondbacks": "D-backs",
    "Atlanta Braves": "Braves", "Baltimore Orioles": "Orioles",
    "Boston Red Sox": "Red Sox", "Chicago Cubs": "Cubs",
    "Chicago White Sox": "White Sox", "Cincinnati Reds": "Reds",
    "Cleveland Guardians": "Guardians", "Colorado Rockies": "Rockies",
    "Detroit Tigers": "Tigers", "Houston Astros": "Astros",
    "Kansas City Royals": "Royals", "Los Angeles Angels": "Angels",
    "Los Angeles Dodgers": "Dodgers", "Miami Marlins": "Marlins",
    "Milwaukee Brewers": "Brewers", "Minnesota Twins": "Twins",
    "New York Mets": "Mets", "New York Yankees": "Yankees",
    "Oakland Athletics": "Athletics", "Athletics": "Athletics",
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

def load_historical_games():
    """从 CSV 和 parquet 收集所有具有 home_score/away_score 的比赛，返回 DataFrame"""
    games = []
    # 先尝试 CSV
    csv_file = "data/historical_predictions.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df = df.dropna(subset=['home_score', 'away_score'])
            if not df.empty:
                games.append(df[['game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score']])
    # 从 parquet 收集
    hist_dir = "data/historical"
    if os.path.exists(hist_dir):
        for f in os.listdir(hist_dir):
            if f.endswith('.parquet'):
                pdf = pd.read_parquet(os.path.join(hist_dir, f))
                if 'home_score' in pdf.columns and 'away_score' in pdf.columns:
                    pdf = pdf.dropna(subset=['home_score', 'away_score'])
                    if not pdf.empty:
                        games.append(pdf[['game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score']])
    if not games:
        return pd.DataFrame()
    all_games = pd.concat(games, ignore_index=True)
    all_games = all_games.drop_duplicates(subset=['game_id'])
    # 统一队名
    all_games['home_team'] = all_games['home_team'].map(TEAM_NAME_MAP).fillna(all_games['home_team'])
    all_games['away_team'] = all_games['away_team'].map(TEAM_NAME_MAP).fillna(all_games['away_team'])
    if 'game_date' in all_games.columns:
        all_games = all_games.sort_values('game_date')
    return all_games

def rebuild():
    df = load_historical_games()
    if df.empty:
        print("没有找到任何具有比分的比赛数据，无法重建")
        return

    # 初始化评级
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    elo_ratings = {team: 1500.0 for team in all_teams}
    league = Glicko2League()
    for team in all_teams:
        league.add_team(team, rating=1500.0, rd=350.0, vol=0.06)

    rated_ids = set()
    processed = 0
    skipped_missing = 0
    skipped_duplicate = 0

    for _, row in df.iterrows():
        gid = str(row.get('game_id'))
        if not gid or gid in {'None', 'nan'}:
            continue
        if gid in rated_ids:
            skipped_duplicate += 1
            continue

        home = row['home_team']
        away = row['away_team']
        try:
            home_score = int(row['home_score'])
            away_score = int(row['away_score'])
        except (ValueError, TypeError):
            skipped_missing += 1
            continue

        # Elo 更新
        simple_elo_update(elo_ratings, home, away, home_score, away_score)

        # Glicko2 更新
        if home not in league.teams:
            league.add_team(home)
        if away not in league.teams:
            league.add_team(away)
        home_team = league.teams[home]
        away_team = league.teams[away]
        # 赛前快照
        home_opp = copy.deepcopy(away_team)
        away_opp = copy.deepcopy(home_team)
        if home_score > away_score:
            home_team.update(home_opp, 1.0)
            away_team.update(away_opp, 0.0)
        elif home_score < away_score:
            home_team.update(home_opp, 0.0)
            away_team.update(away_opp, 1.0)
        else:
            home_team.update(home_opp, 0.5)
            away_team.update(away_opp, 0.5)

        rated_ids.add(gid)
        processed += 1

    # 保存结果
    save_elo_ratings(elo_ratings)
    save_glicko2_league(league)
    with open('data/rated_game_ids.json', 'w') as f:
        json.dump(list(rated_ids), f)

    # 健康报告
    elo_values = list(elo_ratings.values())
    glicko_values = [t.rating for t in league.teams.values()]
    print("===== Rating Rebuild Report =====")
    print(f"Processed games with scores: {processed}")
    print(f"Skipped (missing scores): {skipped_missing}")
    print(f"Skipped (duplicate game_id): {skipped_duplicate}")
    print(f"Total rated game_ids saved: {len(rated_ids)}")
    print(f"Number of teams: {len(all_teams)}")
    print(f"Elo   min: {min(elo_values):.2f}, max: {max(elo_values):.2f}, range: {max(elo_values)-min(elo_values):.2f}")
    print(f"Glicko2 min: {min(glicko_values):.2f}, max: {max(glicko_values):.2f}, range: {max(glicko_values)-min(glicko_values):.2f}")
    print("Team list:", sorted(all_teams))
    print("=================================")

if __name__ == '__main__':
    rebuild()
