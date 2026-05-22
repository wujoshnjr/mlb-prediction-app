import pandas as pd
import numpy as np
from scipy.optimize import minimize

def get_bradley_terry_strengths():
    """使用MLE估计Bradley-Terry模型得到球队强度"""
    history_file = "data/historical_predictions.csv"
    if not os.path.exists(history_file):
        return {}
    df = pd.read_csv(history_file)
    df = df.dropna(subset=['home_win'])
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    team_to_idx = {t:i for i,t in enumerate(teams)}
    n_teams = len(teams)

    def neg_log_likelihood(params):
        strengths = params
        ll = 0
        for _, row in df.iterrows():
            h = team_to_idx.get(row['home_team'])
            a = team_to_idx.get(row['away_team'])
            if h is None or a is None: continue
            diff = strengths[h] - strengths[a]
            prob = 1 / (1 + np.exp(-diff))
            y = row['home_win']
            ll += y * np.log(prob + 1e-10) + (1-y) * np.log(1-prob + 1e-10)
        return -ll

    init = np.zeros(n_teams)
    res = minimize(neg_log_likelihood, init, method='L-BFGS-B')
    strengths = {teams[i]: res.x[i] for i in range(n_teams)}
    return strengths
