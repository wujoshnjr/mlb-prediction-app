import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

HISTORY_FILE = "data/historical_predictions.csv"
REPORT_FILE = "data/backtest_report.csv"

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

    # 分数凯利对比
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

    # Brier / LogLoss
    clean = df[['home_win','pred_home_win']].dropna()
    if len(clean) > 0:
        brier = brier_score_loss(clean['home_win'], clean['pred_home_win'])
        logloss = log_loss(clean['home_win'], clean['pred_home_win'])
        print(f"整体 Brier Score: {brier:.4f} | Log Loss: {logloss:.4f}")

    # ========== 可靠性曲线数据 ==========
    print("\n--- 可靠性曲线 (Reliability Curve) ---")
    prob_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(prob_bins)-1):
        mask = (df['pred_home_win'] >= prob_bins[i]) & (df['pred_home_win'] < prob_bins[i+1])
        if mask.sum() == 0:
            continue
        avg_pred = df.loc[mask, 'pred_home_win'].mean()
        avg_actual = df.loc[mask, 'home_win'].mean()
        print(f"  预测区间 [{prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}]: 平均预测={avg_pred:.3f}, 实际胜率={avg_actual:.3f}, 样本数={mask.sum()}")

    # ========== 分月校准监控 ==========
    print("\n--- 分月 Brier Score ---")
    if 'game_date' in df.columns:
        df['month'] = pd.to_datetime(df['game_date']).dt.month
        for month in sorted(df['month'].dropna().unique()):
            month_df = df[df['month'] == month]
            if len(month_df) > 5:
                month_brier = brier_score_loss(month_df['home_win'], month_df['pred_home_win'])
                print(f"  月份 {int(month)}: Brier={month_brier:.4f}, 样本数={len(month_df)}")
    else:
        print("  缺少 game_date 列，无法分月统计")

    # ========== 重训练建议 ==========
    if brier > 0.25:
        print("\n⚠️ Brier Score 过高，建议立即重新训练模型。")

    # 保存详细报告
    df.to_csv(REPORT_FILE, index=False)
    print(f"\n详细回测报告已保存至 {REPORT_FILE}")

if __name__ == "__main__":
    run_backtest()
