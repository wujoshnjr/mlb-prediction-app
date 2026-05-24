# scripts/lag_features.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_lag_features(home_team, away_team, historical_df, game_date, days=30):
    """
    计算主客队在指定日期前 `days` 天内的胜率差和场均得分差。
    返回 (winrate_diff, runs_diff)，两个均为 float。
    若数据不足或缺失，返回 (0.0, 0.0)。
    """
    if historical_df is None or historical_df.empty:
        return 0.0, 0.0

    try:
        game_date = pd.to_datetime(game_date)
        cutoff = game_date - timedelta(days=days)
        mask = (pd.to_datetime(historical_df['game_date']) >= cutoff) & \
               (pd.to_datetime(historical_df['game_date']) < game_date)
        recent = historical_df[mask]

        if recent.empty:
            return 0.0, 0.0

        # 主队相关比赛
        home_games = recent[(recent['home_team'] == home_team) | (recent['away_team'] == home_team)]
        away_games = recent[(recent['home_team'] == away_team) | (recent['away_team'] == away_team)]

        # 计算胜率（基于 home_win 列）
        if 'home_win' in recent.columns:
            def calc_win_rate(team_games, team):
                if team_games.empty:
                    return 0.5
                wins = 0
                for _, row in team_games.iterrows():
                    if row['home_team'] == team:
                        wins += row['home_win']
                    else:
                        wins += (1 - row['home_win'])
                return wins / len(team_games)
            home_winrate = calc_win_rate(home_games, home_team)
            away_winrate = calc_win_rate(away_games, away_team)
        else:
            home_winrate = 0.5
            away_winrate = 0.5

        # 尝试计算得分差（如果列存在）
        runs_diff = 0.0
        if 'home_score' in recent.columns and 'away_score' in recent.columns:
            home_scores = []
            for _, row in home_games.iterrows():
                if row['home_team'] == home_team:
                    home_scores.append(row['home_score'])
                else:
                    home_scores.append(row['away_score'])
            away_scores = []
            for _, row in away_games.iterrows():
                if row['home_team'] == away_team:
                    away_scores.append(row['home_score'])
                else:
                    away_scores.append(row['away_score'])
            home_runs_avg = np.mean(home_scores) if home_scores else 0.0
            away_runs_avg = np.mean(away_scores) if away_scores else 0.0
            runs_diff = home_runs_avg - away_runs_avg

        winrate_diff = home_winrate - away_winrate
        return round(winrate_diff, 4), round(runs_diff, 2)

    except Exception as e:
        print(f"lag_features error: {e}")
        return 0.0, 0.0
