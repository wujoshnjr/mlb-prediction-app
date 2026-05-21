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
    # 清理空值
    df['home_win'] = df['home_win'].replace('', np.nan)
    df = df.dropna(subset=['home_win'])
    if len(df) == 0:
        print("没有已完成的比赛数据")
        return
    df['home_win'] = df['home_win'].astype(int)

    # 投注判断
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

    # 最大回撤
    cumulative = df['cumulative_profit']
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    # Sharpe Ratio
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

    # CLV 分析（如果数据可用）
    if 'closing_odds' in df.columns and 'home_odds' in df.columns:
        df['closing_odds'] = pd.to_numeric(df['closing_odds'], errors='coerce')
        valid_clv = df[df['closing_odds'].notna() & (df['closing_odds'] > 0)]
        if len(valid_clv) > 0:
            valid_clv['clv'] = valid_clv['home_odds'].astype(float) - valid_clv['closing_odds']
            avg_clv = valid_clv['clv'].mean()
            print("-" * 50)
            print("📈 Closing Line Value (CLV)")
            print(f"有效样本数: {len(valid_clv)}")
            print(f"平均 CLV: {avg_clv:+.4f}")
            if avg_clv > 0.02:
                print("✅ 模型持续拿到优于收盘赔率的价格（长期可能盈利）")
            elif avg_clv < -0.02:
                print("⚠️ 模型赔率不如收盘赔率，需要优化")
            else:
                print("➖ CLV 接近零，市场效率较高")

    print("=" * 50)

if __name__ == "__main__":
    run_backtest()
