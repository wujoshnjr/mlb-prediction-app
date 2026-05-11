"""
MLB 棒球預測模型訓練腳本 (使用 MLB 官方 Stats API，結構正確版)
"""
import requests
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 輔助函數：呼叫 MLB Stats API（正確解析 splits 結構）
# ============================================================
BASE = "https://statsapi.mlb.com/api/v1"

def get_team_stats(year, group):
    """
    group: 'hitting' 或 'pitching'
    回傳 DataFrame，包含所有球隊在該年度的傳統/進階數據
    """
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
    # stats 是一個 list，每個元素可能包含不同 type/group 的 splits
    for entry in data.get('stats', []):
        # 確認是我們要的群組和球季類型
        if entry.get('group', {}).get('displayName') != group:
            continue
        if entry.get('type', {}).get('displayName') != 'season':
            continue

        # 正確的數據在 splits 裡面
        for split in entry.get('splits', []):
            team_name = split['team']['name']
            s = split['stat']

            if group == 'hitting':
                rows.append({
                    'Team': team_name,
                    'AVG': float(s.get('avg', 0)),
                    'OBP': float(s.get('obp', 0)),
                    'SLG': float(s.get('slg', 0)),
                    'OPS': float(s.get('ops', 0)),
                    'wOBA': float(s.get('woba', 0)) if s.get('woba') else np.nan,
                    'wRC+': float(s.get('wrcPlus', 0)) if s.get('wrcPlus') else np.nan,
                })
            else:  # pitching
                ip = float(s.get('inningsPitched', 0))
                k = float(s.get('strikeOuts', 0))
                bb = float(s.get('baseOnBalls', 0))
                hr = float(s.get('homeRuns', 0))
                rows.append({
                    'Team': team_name,
                    'ERA': float(s.get('era', 0)),
                    'WHIP': float(s.get('whip', 0)),
                    'FIP': float(s.get('fip', 0)) if s.get('fip') else np.nan,
                    'K/9': round(k / ip * 9, 2) if ip > 0 else 0,
                    'BB/9': round(bb / ip * 9, 2) if ip > 0 else 0,
                    'HR/9': round(hr / ip * 9, 2) if ip > 0 else 0,
                })

    if not rows:
        raise ValueError(f"找不到 {year} 年 {group} 數據 (可能賽季尚未開始)")

    return pd.DataFrame(rows)

def get_standings(year):
    """取得戰績（MLB Stats API）"""
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
    print(f"⚠️ {year} 年數據取得失敗 ({e})，改用 2024 年數據")
    year = 2024
    bat = get_team_stats(year, 'hitting')
    pitch = get_team_stats(year, 'pitching')

standings_records = get_standings(year)

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
standings_df = pd.DataFrame(standings_records)
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
