# scripts/backtest.py
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

# 确保可以导入项目根目录下的模块（如 config）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HISTORY_FILE = "data/historical_predictions.csv"
REPORT_FILE = "data/backtest_report.csv"
FEATURE_IMPORTANCE_LOG = "data/feature_importance.csv"

# 导入配置（防御性）
try:
    import config
except:
    class config:
        WALKFORWARD_STRICT = False

# 导入 Walk-Forward 模块（防御性）
try:
    from scripts.walkforward import walkforward_train_evaluate
except:
    walkforward_train_evaluate = None

# 模型构建函数（防御性，用于 Walk-Forward）
def _create_model_for_wf():
    """创建一个未训练的 Stacking 模型，与 train_ensemble 保持一致"""
    try:
        from train_ensemble import build_base_estimators, build_meta_learner
        from sklearn.ensemble import StackingClassifier
        from sklearn.model_selection import TimeSeriesSplit
        base = build_base_estimators()
        meta = build_meta_learner()
        model = StackingClassifier(
            estimators=base,
            final_estimator=meta,
            cv=TimeSeriesSplit(n_splits=3),
            stack_method='predict_proba'
        )
        return model
    except Exception as e:
        print(f"无法构建 Walk-Forward 模型: {e}")
        return None

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
        df_no_odds['pred_home_win'] = df_no_odds['manual_no_odds_pred']
        print("\n📉 无赔率策略 (Blind Test)")
        no_odds = evaluate(df_no_odds)
        print_metrics(no_odds, "无赔率")

    # ========== Walk-Forward 赛季评估（原有） ==========
    if 'game_date' in df.columns:
        df['season'] = pd.to_datetime(df['game_date']).dt.year
        seasons = sorted(df['season'].unique())
        if len(seasons) >= 3:
            print("\n📅 Walk-Forward 赛季评估")
            for i, test_season in enumerate(seasons[2:], start=2):
                train_seasons = seasons[:i]
                test_df = df[df['season'] == test_season]
                if len(test_df) < 10:
                    continue
                metrics = evaluate(test_df)
                print(f"  训练: {train_seasons}, 测试: {int(test_season)} → 比赛数={metrics['total']}, ROI={metrics['roi']:.2%}, 胜率={metrics['win_rate']:.2%}, Brier={metrics['brier']:.4f}")

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

    # ========== CLV 跟踪 ==========
    if 'closing_odds' in df.columns and df['closing_odds'].notna().sum() > 0:
        df_clv = df.dropna(subset=['closing_odds'])
        df_clv['clv'] = df_clv['home_odds'] - df_clv['closing_odds']
        avg_clv = df_clv['clv'].mean()
        beat_rate = (df_clv['clv'] > 0).mean()
        print(f"\n💹 收盘盘口价值 (CLV)")
        print(f"  平均 CLV: {avg_clv:+.4f}")
        print(f"  击败收盘盘口比例: {beat_rate:.1%}")

    # ========== 特征稳定性 ==========
    if os.path.exists(FEATURE_IMPORTANCE_LOG):
        fi_df = pd.read_csv(FEATURE_IMPORTANCE_LOG)
        if len(fi_df) >= 2:
            print("\n🔍 特征重要性漂移检查 (最近两次训练对比)")
            last = fi_df.iloc[-1]
            prev = fi_df.iloc[-2]
            diffs = {}
            for col in fi_df.columns[:-1]:
                if col in last and col in prev:
                    diffs[col] = abs(last[col] - prev[col])
            sorted_diffs = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
            for feat, d in sorted_diffs[:5]:
                print(f"  {feat}: 变化 {d:.4f}")

    # ========== 资金蒙特卡洛 ==========
    print("\n🎲 资金曲线蒙特卡洛模拟 (1000次 Bootstrap)")
    if 'profit_0.25' in df.columns:
        monte_carlo_bankroll(df['profit_0.25'].values)
    else:
        profit_series = []
        for _, row in df.iterrows():
            if 'Bet' not in str(row.get('ml_rec', '')):
                profit_series.append(0)
            else:
                odds = row.get('home_odds', 2.0)
                if pd.isna(odds) or odds <= 1:
                    odds = 2.0
                kelly_f = max(0, row['pred_home_win'] - (1 - row['pred_home_win']) / (odds - 1)) * 0.25
                profit = kelly_f * (odds - 1) if row['home_win'] == 1 else -kelly_f
                profit_series.append(profit)
        monte_carlo_bankroll(np.array(profit_series))

    df.to_csv(REPORT_FILE, index=False)
    print(f"\n详细报告已保存至 {REPORT_FILE}")

    # ========== 严格 Walk-Forward 训练评估（新增） ==========
    if config.WALKFORWARD_STRICT:
        print("\n" + "=" * 60)
        print("🚀 严格 Walk-Forward 训练与评估")
        print("=" * 60)
        if walkforward_train_evaluate is None:
            print("  ❌ Walk-Forward 模块未正确导入，跳过。")
        else:
            train_file = "data/training_features.parquet"
            if not os.path.exists(train_file):
                print(f"  ❌ 训练特征文件 {train_file} 不存在，无法执行 Walk-Forward。")
                print("     请先运行特征工程生成该文件。")
            else:
                try:
                    data = pd.read_parquet(train_file)
                    target_col = 'target'
                    date_col = 'date'
                    if target_col not in data.columns or date_col not in data.columns:
                        print("  ❌ 训练数据缺少 'target' 或 'date' 列。")
                    else:
                        feature_cols = [c for c in data.columns if c not in [target_col, date_col, 'game_id']]
                        X = data[feature_cols]
                        y = data[target_col]
                        dates = pd.to_datetime(data[date_col])

                        model_builder = _create_model_for_wf
                        if model_builder is None:
                            print("  ❌ 无法创建模型构建函数。")
                        else:
                            print("  开始逐期训练与评估...")
                            y_true, y_pred, eval_dates = walkforward_train_evaluate(
                                X, y, dates, model_builder, gap_days=7, test_days=1,
                                train_min_days=180, use_optuna=False
                            )
                            brier = brier_score_loss(y_true, y_pred)
                            ll = log_loss(y_true, y_pred)
                            acc = np.mean((y_pred > 0.5) == y_true)
                            roi = 2 * acc - 1
                            print(f"  Walk-Forward 结果 (比赛数={len(y_true)}):")
                            print(f"    准确率: {acc:.2%}, ROI: {roi:.2%}, Brier: {brier:.4f}, LogLoss: {ll:.4f}")
                except Exception as e:
                    print(f"  ❌ Walk-Forward 评估失败: {e}")

def evaluate(df):
    df = df.copy()
    df['bet'] = df['ml_rec'].apply(lambda x: 1 if 'Bet' in str(x) else 0)
    def profit_kelly(row, fraction=0.25):
        if row['bet'] == 0: return 0
        odds = row.get('home_odds', 2.0)
        if pd.isna(odds) or odds <= 1: odds = 2.0
        kelly_f = max(0, row['pred_home_win'] - (1 - row['pred_home_win']) / (odds - 1)) * fraction
        return (kelly_f * (odds - 1)) if row['home_win'] == 1 else -kelly_f
    df['profit'] = df.apply(profit_kelly, axis=1)
    total = len(df)
    bets = df['bet'].sum()
    profit = df['profit'].sum()
    roi = profit / bets if bets > 0 else 0
    win_rate = df[df['bet'] == 1]['home_win'].mean() if bets > 0 else 0
    clean = df[['home_win','pred_home_win']].dropna()
    brier = brier_score_loss(clean['home_win'], clean['pred_home_win']) if len(clean) > 0 else 0
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
