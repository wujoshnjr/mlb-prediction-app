"""
MLB 棒球預測模型訓練腳本
功能：抓取 MLB 數據 → 特徵工程 → 訓練 XGBoost 模型 → 輸出預測結果
"""

import pandas as pd
import numpy as np
from pybaseball import team_batting, team_pitching, standings
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 資料蒐集
# ============================================================
print("📡 正在抓取 MLB 團隊數據...")

# 取得 2025 年團隊打擊與投球數據（若當前賽季尚未開始，改抓 2024 年）
try:
    bat = team_batting(2025)
    pitch = team_pitching(2025)
    if bat.empty or pitch.empty:
        raise ValueError("2025 數據為空，改用 2024")
except:
    print("⚠️ 2025 數據尚不可用，改用 2024 年數據")
    bat = team_batting(2024)
    pitch = team_pitching(2024)

# 取得戰績排名（含勝場、敗場）
try:
    st = standings(2025)
    if not st:
        raise ValueError("2025 排名為空，改用 2024")
except:
    st = standings(2024)

print(f"✅ 數據抓取完成：打擊 {bat.shape[0]} 筆，投球 {pitch.shape[0]} 筆")

# ============================================================
# 2. 特徵工程
# ============================================================
print("🔧 正在進行特徵工程...")

# 合併打擊與投球數據
df = bat.merge(pitch, on='Team', suffixes=('_bat', '_pitch'))

# 挑選重要特徵
feature_cols = [
    'AVG_bat', 'OBP_bat', 'SLG_bat', 'OPS_bat',      # 打擊指標
    'wOBA_bat', 'wRC+_bat',                            # 進階打擊
    'ERA_pitch', 'WHIP_pitch', 'FIP_pitch',            # 投手指標
    'K/9_pitch', 'BB/9_pitch', 'HR/9_pitch'           # 投手進階
]

# 只保留有完整數據的欄位
available_cols = [col for col in feature_cols if col in df.columns]
df = df[['Team'] + available_cols].dropna()

print(f"✅ 特徵工程完成，可用特徵數：{len(available_cols)}")
print(f"   使用特徵：{available_cols}")

# ============================================================
# 3. 建立訓練標籤（簡化版：用勝率當標籤）
# ============================================================
# 從 standings 取得勝率
records = []
for league in st:
    for team in league:
        records.append({
            'Team': team.name,
            'W': team.W,
            'L': team.L,
            'WinPct': team.W / (team.W + team.L) if (team.W + team.L) > 0 else 0.5
        })

standings_df = pd.DataFrame(records)
df = df.merge(standings_df[['Team', 'WinPct']], on='Team')

# 標籤：勝率 > 0.500 為強隊 (1)，否則為弱隊 (0)
df['is_strong'] = (df['WinPct'] > 0.500).astype(int)

print(f"   強隊數量：{df['is_strong'].sum()}，弱隊數量：{len(df) - df['is_strong'].sum()}")

# ============================================================
# 4. 訓練模型
# ============================================================
print("🧠 正在訓練 XGBoost 模型...")

X = df[available_cols]
y = df['is_strong']

# 使用時間序列交叉驗證
tscv = TimeSeriesSplit(n_splits=3)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 交叉驗證
scores = []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

print(f"✅ 交叉驗證準確率：{np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

# 最終訓練（用全部數據）
model.fit(X, y)

# ============================================================
# 5. 儲存模型與預測結果
# ============================================================
print("💾 正在儲存模型與預測結果...")

# 儲存模型
joblib.dump(model, 'outputs/xgb_model.pkl')

# 產出所有球隊的預測機率
df['predicted_strong_prob'] = model.predict_proba(X)[:, 1]
df['predicted_label'] = model.predict(X)

# 儲存 CSV
output_df = df[['Team', 'WinPct', 'predicted_strong_prob', 'predicted_label'] + available_cols]
output_df = output_df.rename(columns={
    'Team': 'team',
    'WinPct': 'actual_win_pct',
    'predicted_strong_prob': 'model_strength_prob',
    'predicted_label': 'predicted_strong'
})
output_df.to_csv('outputs/predictions.csv', index=False)

print("🎉 全部完成！預測結果已儲存至 outputs/predictions.csv")
print(output_df[['team', 'actual_win_pct', 'model_strength_prob']].head(10).to_string(index=False))
