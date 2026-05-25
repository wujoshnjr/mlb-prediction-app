# scripts/backtest.py
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HISTORY_FILE = "data/historical_predictions.csv"
REPORT_FILE = "data/backtest_report.csv"
FEATURE_IMPORTANCE_LOG = "data/feature_importance.csv"

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

    # ========== 整体评估 ==========
    print("=" * 60)
    print("📊 整体回测报告")
    print("=" * 60)
    overall = evaluate(df)
    print_metrics(overall, "整体")

    # ========== 无赔率盲测 ==========
    if 'manual_no_odds_pred' in df.columns:
        df_no_odds = df.copy()
        df_no_odds['predicted_home_win_pct'] = df_no_odds['manual_no_odds_pred']
        print("\n📉 无赔率策略 (Blind Test)")
        no_odds = evaluate(df_no_odds)
        print_metrics(no_odds, "无赔率")

    # ========== 分月 ROI ==========
    if 'game_date' in df.columns:
        df['month'] = pd.to_datetime(df['game_date']).dt.month
        print("\n📅 分月 ROI 分析")
        for month in sorted(df['month'].unique()):
            month_df = df[df['month'] == month]
            if len(month_df) < 5:
                continue
            m = evaluate(month_df)
            print(f"  月份 {int(month)}: 比赛={m['total']}, 投注={m['bets']}, ROI={m['roi']:.2%}, 胜率={m['win_rate']:.2%}, Brier={m['brier']:.4f}")

    # ========== 资金蒙特卡洛 ==========
    print("\n🎲 资金曲线蒙特卡洛模拟 (1000次 Bootstrap)")
    profit_series = []
    for _, row in df.iterrows():
        if 'Bet' not in str(row.get('moneyline_recommendation', '')):
            profit_series.append(0)
        else:
            odds = row.get('home_odds', 2.0)
            if pd.isna(odds) or odds <= 1:
                odds = 2.0
            kelly_f = max(0, row['predicted_home_win_pct'] - (1 - row['predicted_home_win_pct']) / (odds - 1)) * 0.25
            profit = kelly_f * (odds - 1) if row['home_win'] == 1 else -kelly_f
            profit_series.append(profit)
    monte_carlo_bankroll(np.array(profit_series))

    df.to_csv(REPORT_FILE, index=False)
    print(f"\n详细报告已保存至 {REPORT_FILE}")

def evaluate(df):
    df = df.copy()
    df['bet'] = df['moneyline_recommendation'].apply(lambda x: 1 if 'Bet' in str(x) else 0)
    def profit_kelly(row, fraction=0.25):
        if row['bet'] == 0: return 0
        odds = row.get('home_odds', 2.0)
        if pd.isna(odds) or odds <= 1: odds = 2.0
        kelly_f = max(0, row['predicted_home_win_pct'] - (1 - row['predicted_home_win_pct']) / (odds - 1)) * fraction
        return (kelly_f * (odds - 1)) if row['home_win'] == 1 else -kelly_f
    df['profit'] = df.apply(profit_kelly, axis=1)
    total = len(df)
    bets = df['bet'].sum()
    profit = df['profit'].sum()
    roi = profit / bets if bets > 0 else 0
    win_rate = df[df['bet'] == 1]['home_win'].mean() if bets > 0 else 0
    clean = df[['home_win', 'predicted_home_win_pct']].dropna()
    brier = brier_score_loss(clean['home_win'], clean['predicted_home_win_pct']) if len(clean) > 0 else 0
    max_dd = (df['profit'].cumsum() - df['profit'].cumsum().cummax()).min()
    return {'total': total, 'bets': bets, 'profit': profit, 'roi': roi, 'win_rate': win_rate, 'brier': brier, 'max_dd': max_dd}

def print_metrics(m, label):
    print(f"  {label}: 比赛={m['total']}, 投注={m['bets']}, ROI={m['roi']:.2%}, 胜率={m['win_rate']:.2%}, Brier={m['brier']:.4f}, MaxDD={m['max_dd']:.2f}")

def monte_carlo_bankroll(profit_series, initial=10000, n_sim=1000, ruin_threshold=0.2):
    np.random.seed(42)
    finals = []
    max_dds = []
    for _ in range(n_sim):
        boot = np.random.choice(profit_series, size=len(profit_series), replace=True)
        curve = initial + np.cumsum(boot)
        finals.append(curve[-1])
        max_dds.append((np.maximum.accumulate(curve) - curve).max())
    finals = np.array(finals)
    ruin = np.mean(finals < initial * ruin_threshold)
    expected_final = np.mean(finals)
    prob_double = np.mean(finals >= initial * 2)
    print(f"  初始资金: {initial}")
    print(f"  期望最终资金: {expected_final:.0f}")
    print(f"  破产概率 (<{int(initial*ruin_threshold)}): {ruin:.2%}")
    print(f"  翻倍概率: {prob_double:.2%}")
    print(f"  最大回撤 (95%): {np.percentile(max_dds, 95):.0f}")

if __name__ == "__main__":
    run_backtest()
