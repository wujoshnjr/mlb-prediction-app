import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

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

    def calc_profit(row):
        if row['bet'] == 0:
            return 0
        odds = row.get('home_odds', 2.0)
        if pd.isna(odds) or odds <= 1:
            odds = 2.0
        if row['home_win'] == 1:
            return odds - 1
        else:
            return -1

    df['profit'] = df.apply(calc_profit, axis=1)
    df['cumulative_profit'] = df['profit'].cumsum()

    total_bets = df['bet'].sum()
    total_profit = df['profit'].sum()
    roi = total_profit / total_bets if total_bets > 0 else 0
    win_rate = df[df['bet'] == 1]['home_win'].mean() if total_bets > 0 else 0

    cumulative = df['cumulative_profit']
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    bet_profits = df[df['bet'] == 1]['profit']
    if len(bet_profits) > 1 and bet_profits.std() > 0:
        sharpe = bet_profits.mean() / bet_profits.std() * np.sqrt(len(bet_profits))
    else:
        sharpe = 0

    print("=" * 50)
    print("📊 回测报告")
    print("=" * 50)
    print(f"总比赛数: {len(df)}")
    print(f"总投注数: {total_bets}")
    print(f"总盈利: {total_profit:.2f} 单位")
    print(f"ROI: {roi:.2%}")
    print(f"胜率: {win_rate:.2%}")
    print(f"最大回撤: {max_drawdown:.2f} 单位")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    run_backtest()
