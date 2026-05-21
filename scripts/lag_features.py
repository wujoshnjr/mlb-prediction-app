"""
滞后特征生成器
计算球队最近7/14/30天的滚动指标
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_lag_features(home_team, away_team, historical_games_df, date_str, days=30):
    """
    historical_games_df: 包含 home_team, away_team, home_score, away_score, date 的历史比赛数据
    返回: home_winrate_30d, away_winrate_30d, home_avg_runs_30d, away_avg_runs_30d
    """
    if historical_games_df is None or historical_games_df.empty:
        return 0.5, 0.5, 4.5, 4.5

    current_date = datetime.strptime(date_str, "%Y-%m-%d")
    cutoff = current_date - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    # 筛选最近N天的比赛
    recent_games = historical_games_df[historical_games_df['date'] >= cutoff_str]

    # 主队近N战胜率及场均得分
    home_games = recent_games[
        (recent_games['home_team'] == home_team) | (recent_games['away_team'] == home_team)
    ]
    home_wins = len(home_games[
        ((home_games['home_team'] == home_team) & (home_games['home_score'] > home_games['away_score'])) |
        ((home_games['away_team'] == home_team) & (home_games['away_score'] > home_games['home_score']))
    ])
    home_winrate = home_wins / len(home_games) if len(home_games) > 0 else 0.5
    home_runs_list = []
    for _, g in home_games.iterrows():
        if g['home_team'] == home_team:
            home_runs_list.append(g['home_score'])
        else:
            home_runs_list.append(g['away_score'])
    home_avg_runs = np.mean(home_runs_list) if home_runs_list else 4.5

    # 客队近N战胜率及场均得分
    away_games = recent_games[
        (recent_games['home_team'] == away_team) | (recent_games['away_team'] == away_team)
    ]
    away_wins = len(away_games[
        ((away_games['home_team'] == away_team) & (away_games['home_score'] > away_games['away_score'])) |
        ((away_games['away_team'] == away_team) & (away_games['away_score'] > away_games['home_score']))
    ])
    away_winrate = away_wins / len(away_games) if len(away_games) > 0 else 0.5
    away_runs_list = []
    for _, g in away_games.iterrows():
        if g['home_team'] == away_team:
            away_runs_list.append(g['home_score'])
        else:
            away_runs_list.append(g['away_score'])
    away_avg_runs = np.mean(away_runs_list) if away_runs_list else 4.5

    return home_winrate, away_winrate, home_avg_runs, away_avg_runs
