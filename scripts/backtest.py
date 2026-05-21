import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

HISTORY_FILE = "data/historical_predictions.csv"

def run_backtest():
    if not os.path.exists(HISTORY_FILE):
        print("历史数据文件不存在")
        return
    df = pd.read_csv(HISTORY_FILE)
    df['home_win'] = df['home_win'].replace('', np.nan)
    df = df.dropna(subset=['home_win'])
    if len(df) == 0:
        print("没有已完成的比赛数据")
        return
    df['home_win'] = df['home_win'].astype(int)
    df['bet'] = df['ml_rec'].apply(lambda x: 1 if 'Bet' in str(x) else 0)

    def profit_kelly(row, fraction=0.25):
        if row['bet'] == 0: return 0
        odds = row.get('home_odds', 2.0)
        if pd.isna(odds) or odds <= 1: odds = 2.0
        kelly_f = max(0, row['pred_home_win'] - (1 - row['pred_home_win']) / (odds - 1)) * fraction
        if row['home_win'] == 1:
            return kelly_f * (odds - 1)
        else:
            return -kelly_f

    for frac, label in [(0.25, "1/4"), (0.5, "1/2")]:
        df[f'profit_{frac}'] = df.apply(lambda r: profit_kelly(r, fraction=frac), axis=1)
        cum = df[f'profit_{frac}'].cumsum()
        total_bets = df['bet'].sum()
        total_profit = df[f'profit_{frac}'].sum()
        roi = total_profit / total_bets if total_bets > 0 else 0
        max_dd = (cum - cum.cummax()).min()
        print(f"--- {label} Kelly ---")
        print(f"ROI: {roi:.2%}, 总盈利: {total_profit:.2f}, 最大回撤: {max_dd:.2f}")

    clean = df[['home_win','pred_home_win']].dropna()
    if len(clean) > 0:
        brier = brier_score_loss(clean['home_win'].astype(int), clean['pred_home_win'].astype(float))
        logloss = log_loss(clean['home_win'].astype(int), clean['pred_home_win'].astype(float))
        print(f"Brier Score: {brier:.4f} | Log Loss: {logloss:.4f}")
        if brier > 0.25:
            print("\n⚠️ Brier Score 过高，建议立即重新训练模型。")

if __name__ == "__main__":
    run_backtest()
