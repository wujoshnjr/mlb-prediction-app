# scripts/feature_selection.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

MODEL_PATH = "data/calibrator.pkl"
if not os.path.exists(MODEL_PATH):
    print("模型文件不存在，请先训练模型")
    sys.exit(1)

model = joblib.load(MODEL_PATH)

EXPECTED_FEATURES = [
    'elo_diff',
    'sp_era_diff', 'sp_fip_diff', 'sp_stuff_plus_diff', 'sp_csw_diff',
    'bullpen_ip_diff', 'rest_diff',
    'dynamic_park_factor',
    'platoon_ops_diff', 'statcast_launch_speed_diff', 'statcast_barrel_diff',
    'statcast_hard_hit_diff', 'statcast_woba_diff',
    'timezone_diff', 'is_day_game', 'back2back_diff',
    'catcher_era_diff', 'cs_diff', 'wind_effect',
    'temp_effect', 'precip_effect', 'injury_diff',
    'dynamic_pythag_diff', 'log5_prob', 'lag30_winrate_diff', 'lag30_runs_diff',
    'pitch_movement_diff',
    'k_pct_diff', 'bb_pct_diff', 'avg_bat_speed_diff',
    'pitcher_rating_diff', 'odds_change', 'odds_momentum',
    'zone_size', 'k_rate', 'bullpen_availability_diff',
    'elo_momentum_7d', 'elo_momentum_30d', 'barrel_pa_diff', 'hardhit_pa_diff',
    'swing_miss_diff', 'csw_diff', 'barrel_bb_pct_diff',
    'sprint_speed_diff', 'pitch_type_matchup_score',
    'top3_woba_diff', 'winrate_diff', 'bt_strength_diff',
    'home_usage_magnitude', 'away_usage_magnitude',
    'home_shift_score', 'away_shift_score',
    'home_delta_FF', 'home_delta_SL', 'home_delta_CH', 'home_delta_CU',
    'home_delta_FC', 'home_delta_SI', 'home_delta_KC', 'home_delta_FS',
    'away_delta_FF', 'away_delta_SL', 'away_delta_CH', 'away_delta_CU',
    'away_delta_FC', 'away_delta_SI', 'away_delta_KC', 'away_delta_FS'
]

HISTORY_FILE = "data/historical_predictions.csv"
if not os.path.exists(HISTORY_FILE):
    print("历史预测文件不存在")
    sys.exit(1)

df = pd.read_csv(HISTORY_FILE)
df['home_win'] = df['home_win'].replace('', np.nan)
df = df.dropna(subset=['home_win'])
if len(df) < 100:
    print("数据量不足")
    sys.exit(1)

for col in EXPECTED_FEATURES:
    if col not in df.columns:
        df[col] = 0.0
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

X = df[EXPECTED_FEATURES].values

np.random.seed(42)
bg_idx = np.random.choice(len(X), min(500, len(X)), replace=False)
X_bg = X[bg_idx]
sample_idx = np.random.choice(len(X), min(200, len(X)), replace=False)
X_sample = X[sample_idx]

print("开始计算 SHAP 值（可能需要几分钟）...")
explainer = shap.KernelExplainer(model.predict_proba, X_bg)
shap_values = explainer.shap_values(X_sample, nsamples=100)

if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': EXPECTED_FEATURES,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print("\n======== 特征重要性排名 (Top 20) ========")
print(importance_df.head(20).to_string(index=False))
print("\n======== 建议移除的低重要性特征 (Bottom 10) ========")
print(importance_df.tail(10).to_string(index=False))

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_vals, X_sample, feature_names=EXPECTED_FEATURES, show=False)
plt.tight_layout()
plt.savefig("report/shap_summary.png", dpi=150)
print("\nSHAP 汇总图已保存至 report/shap_summary.png")

importance_df.to_csv("data/shap_importance.csv", index=False)
print("重要性报告已保存至 data/shap_importance.csv")
