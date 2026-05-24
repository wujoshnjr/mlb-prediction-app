# scripts/lag_features.py
import pandas as pd
import numpy as np
from datetime import timedelta

def calculate_lag_features(home_team, away_team, historical_df, game_date, days=30):
    """
    返回 (winrate_diff, runs_diff)
    若无足够数据，返回 (0.0, 0.0)
    """
    if historical_df is None or historical_df.empty:
        return 0.0, 0.0

    # 自适应日期列名
    if 'game_date' in historical_df.columns:
        date_col = 'game_date'
    elif 'date' in historical_df.columns:
        date_col = 'date'
    else:
        print("lag_features: historical_df 缺少 date/game_date 列")
        return 0.0, 0.0

    try:
        game_date = pd.to_datetime(game_date)
        historical_dates = pd.to_datetime(historical_df[date_col], errors='coerce')
        cutoff = game_date - timedelta(days=days)
        mask = (historical_dates >= cutoff) & (historical_dates < game_date)
        recent = historical_df[mask]
        if recent.empty:
            return 0.0, 0.0

        home_games = recent[(recent['home_team'] == home_team) | (recent['away_team'] == home_team)]
        away_games = recent[(recent['home_team'] == away_team) | (recent['away_team'] == away_team)]

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

        return round(home_winrate - away_winrate, 4), round(runs_diff, 2)
    except Exception as e:
        print(f"lag_features error: {e}")
        return 0.0, 0.0
