"""
MLB 棒球預測模型訓練腳本（使用 MLB 官方 API，完全合法）
"""
import pandas as pd
import numpy as np
from pybaseball import standings, statcast_team_batting, statcast_team_pitching
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 資料蒐集（使用 MLB 官方 Statcast API）
# ============================================================
print("📡 正在抓取 MLB 團隊數據（官方 API）...")

year = 2025
try:
    st = standings(year)
    if not st or len(st) == 0:
        raise ValueError("無排名數據")
except:
    year = 2024
    st = standings(year)

# 用 Statcast 團隊數據（來自 MLB 官方 API，不是 FanGraphs）
bat = statcast_team_batting(year)
pitch = statcast_team_pitching(year)

print(f"✅ 數據抓取完成：打擊 {bat.shape[0]} 筆，投球 {pitch.shape[0]} 筆，年份 {year}")

# ============================================================
# 2. 特徵工程
# ============================================================
print("🔧 正在進行特徵工程...")

# 確保 Team 欄位一致
bat = bat.rename(columns={'team_name': 'Team'})
pitch = pitch.rename(columns={'team_name': 'Team'})

df = bat.merge(pitch, on='Team', suffixes=('_bat', '_pitch'))

# Statcast 團隊數據可能欄位不同，動態擷取可用數值欄位
exclude_cols = ['Team', 'team_id_bat', 'team_id_pitch', 'year_bat', 'year_pitch']
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
available_cols = [col for col in numeric_cols if col not in exclude_cols]

df = df[['Team'] + available_cols].dropna()
print(f"✅ 特徵工程完成，可用特徵數：{len(available_cols)}")

# ============================================================
# 3. 建立訓練標籤
# ============================================================
records = []
for league in st:
    for team in league:
        win_pct = team.W / (team.W + team.L) if (team.W + team.L) > 0 else 0.5
        records.append({
            'Team': team.name,
            'W': team.W,
            'L': team.L,
            'WinPct': win_pct
        })

standings_df = pd.DataFrame(records)
df = df.merge(standings_df[['Team', 'WinPct']], on='Team', how='inner')
df['is_strong'] = (df['WinPct'] > 0.500).astype(int)
print(f"   強隊數量：{df['is_strong'].sum()}，弱隊數量：{len(df) - df['is_strong'].sum()}")

# ============================================================
# 4. 訓練模型
# ============================================================
print("🧠 正在訓練 XGBoost 模型...")

X = df[available_cols]
y = df['is_strong']

model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                      random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# ============================================================
# 5. 儲存模型與預測結果
# ============================================================
print("💾 正在儲存模型與預測結果...")

joblib.dump(model, 'outputs/xgb_model.pkl')
df['predicted_strong_prob'] = model.predict_proba(X)[:, 1]
df['predicted_label'] = model.predict(X)

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
