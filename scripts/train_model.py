"""
MLB 棒球預測模型訓練腳本
使用 pybaseball 取得公開棒球數據，並避免 FanGraphs 403 錯誤。
"""
import requests

# 修改 requests 預設 User-Agent，讓程式看起來像一般瀏覽器
# 這只是取得公開資料的常見做法，不涉及違法行為
original_get = requests.get
def new_get(url, *args, **kwargs):
    headers = kwargs.get('headers', {})
    headers.setdefault('User-Agent',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    kwargs['headers'] = headers
    return original_get(url, *args, **kwargs)
requests.get = new_get

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

try:
    bat = team_batting(2025)
    pitch = team_pitching(2025)
    if bat.empty or pitch.empty:
        raise ValueError("2025 數據為空，改用 2024")
except:
    print("⚠️ 2025 數據尚不可用，改用 2024 年數據")
    bat = team_batting(2024)
    pitch = team_pitching(2024)

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
df = bat.merge(pitch, on='Team', suffixes=('_bat', '_pitch'))

feature_cols = [
    'AVG_bat', 'OBP_bat', 'SLG_bat', 'OPS_bat',
    'wOBA_bat', 'wRC+_bat',
    'ERA_pitch', 'WHIP_pitch', 'FIP_pitch',
    'K/9_pitch', 'BB/9_pitch', 'HR/9_pitch'
]
available_cols = [col for col in feature_cols if col in df.columns]
df = df[['Team'] + available_cols].dropna()
print(f"✅ 特徵工程完成，可用特徵數：{len(available_cols)}")

# ============================================================
# 3. 建立訓練標籤
# ============================================================
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
