"""
MLB 棒球預測模型訓練腳本 (最終穩定版 + 自動建立 outputs)
"""
import requests
import sys
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 輔助函數：呼叫 MLB Stats API
# ============================================================
BASE = "https://statsapi.mlb.com/api/v1"

def get_team_stats(year, group):
    url = f"{BASE}/teams/stats"
    params = {
        "stats": "season",
        "group": group,
        "season": str(year),
        "sportIds": 1,
        "hydrate": "team"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    rows = []
    for entry in data.get('stats', []):
        if entry.get('group', {}).get('displayName') != group:
            continue
        if entry.get('type', {}).get('displayName') != 'season':
            continue
        for split in entry.get('splits', []):
            team_name = split['team']['name']
            stat = split.get('stat', {})
            if not stat:
                continue
            row = {'Team': team_name}
            for k, v in stat.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    pass
            rows.append(row)

    if not rows:
        raise ValueError(f"⚠️ {year} 年 {group} 數據為空，可能賽季尚未開始")
    return pd.DataFrame(rows)

def get_standings(year):
    url = f"{BASE}/standings"
    params = {
        "leagueId": "103,104",
        "season": str(year),
        "hydrate": "team"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    records_list = []
    for record in r.json().get('records', []):
        for team_record in record.get('teamRecords', []):
            t = team_record['team']
            w = team_record['wins']
            l = team_record['losses']
            records_list.append({
                'Team': t['name'],
                'W': w,
                'L': l,
                'WinPct': w / (w + l) if (w + l) > 0 else 0.5
            })
    return records_list

# ============================================================
# 1. 資料蒐集
# ============================================================
print("📡 正在從 MLB Stats API 抓取數據...")
year = 2025

try:
    bat = get_team_stats(year, 'hitting')
    pitch = get_team_stats(year, 'pitching')
except Exception as e:
    print(f"⚠️ {year} 年數據取得失敗: {e}")
    year = 2024
    bat = get_team_stats(year, 'hitting')
    pitch = get_team_stats(year, 'pitching')

standings_records = get_standings(year)
print(f"✅ 數據抓取完成：打擊 {bat.shape}，投球 {pitch.shape}")

# ============================================================
# 2. 特徵工程
# ============================================================
print("🔧 正在進行特徵工程...")
df = bat.merge(pitch, on='Team', suffixes=('_bat', '_pitch'))

exclude = ['Team', 'Team_bat', 'Team_pitch']
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
available_cols = [c for c in numeric_cols if c not in exclude]

if len(available_cols) == 0:
    print("❌ 沒有可用的數值特徵")
    sys.exit(1)

df = df[['Team'] + available_cols].dropna()
print(f"✅ 特徵工程完成，可用特徵數：{len(available_cols)}")

# ============================================================
# 3. 建立訓練標籤
# ============================================================
standings_df = pd.DataFrame(standings_records)
df = df.merge(standings_df[['Team', 'WinPct']], on='Team', how='inner')
df['is_strong'] = (df['WinPct'] > 0.500).astype(int)
print(f"   強隊數量：{df['is_strong'].sum()}，弱隊數量：{len(df) - df['is_strong'].sum()}")

if df.shape[0] < 5:
    print("❌ 資料筆數不足")
    sys.exit(1)

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
# ➕ 自動建立 outputs 資料夾
os.makedirs('outputs', exist_ok=True)

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
