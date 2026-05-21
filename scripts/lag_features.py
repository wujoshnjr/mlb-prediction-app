import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_lag_features(home_team, away_team, historical_games_df, date_str, days=30):
    if historical_games_df is None or historical_games_df.empty:
        return 0.5, 0.5, 4.5, 4.5
    current_date = datetime.strptime(date_str, "%Y-%m-%d")
    cutoff = current_date - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    recent = historical_games_df[historical_games_df['date'] >= cutoff_str]

    def get_stats(team, is_home):
        if is_home:
            games = recent[recent['home_team'] == team]
            wins = len(games[games['home_score'] > games['away_score']])
            runs = games['home_score'].tolist()
        else:
            games = recent[recent['away_team'] == team]
            wins = len(games[games['away_score'] > games['home_score']])
            runs = games['away_score'].tolist()
        winrate = wins / len(games) if len(games) > 0 else 0.5
        avg_runs = np.mean(runs) if runs else 4.5
        return winrate, avg_runs

    home_wr, home_runs = get_stats(home_team, True)
    away_wr, away_runs = get_stats(away_team, False)
    return home_wr, away_wr, home_runs, away_runs
