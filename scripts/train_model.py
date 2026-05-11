"""
MLB 棒球預測模型訓練腳本 (使用 MLB 官方 Stats API，完全合法)
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib
import statsapi
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 輔助函數：從 Stats API 取得團隊數據
# ============================================================
def get_team_batting_stats(year):
    """取得全聯盟團隊打擊數據"""
    data = statsapi.get('teams_stats', {'stats': 'season', 'group': 'hitting', 'season': str(year)})
    rows = []
    for team_data in data['stats']:
        # team_data 是 list，每個元素是一個 team 的 stats
        for t in team_data:
            team_name = t['team']['name']
            stats = t['stats']
            # 可能有多筆 split (e.g. 'total', 'home', 'away')，我們取 'total'
            total_stats = [s for s in stats if s['type'] == 'season' and s['group'] == 'hitting']
            if not total_stats:
                continue
            s = total_stats[0]['stats']
            rows.append({
                'Team': team_name,
                'AVG': float(s.get('avg', 0)),
                'OBP': float(s.get('obp', 0)),
                'SLG': float(s.get('slg', 0)),
                'OPS': float(s.get('ops', 0)),
                'wOBA': float(s.get('woba', 0)) if s.get('woba') else np.nan,
                'wRC+': float(s.get('wrcPlus', 0)) if s.get('wrcPlus') else np.nan,
            })
    return pd.DataFrame(rows)

def get_team_pitching_stats(year):
    """取得全聯盟團隊投球數據"""
    data = statsapi.get('teams_stats', {'stats': 'season', 'group': 'pitching', 'season': str(year)})
    rows = []
    for team_data in data['stats']:
        for t in team_data:
            team_name = t['team']['name']
            stats = t['stats']
            total_stats = [s for s in stats if s['type'] == 'season' and s['group'] == 'pitching']
            if not total_stats:
                continue
            s = total_stats[0]['stats']
            # 計算 K/9, BB/9, HR/9
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
    return pd.DataFrame(rows)

def get_standings(year):
    """利用 statsapi 取得戰績"""
    # statsapi.standings_data() 回傳 dict
    st = statsapi.standings_data(season=year)
    records = []
    for league, divs in st.items():
        for div, teams in divs.items():
            for team in teams:
                records.append({
                    'Team': team['name'],
                    'W': team['w'],
                    'L': team['l'],
                    'WinPct': team['w'] / (team['w'] + team['l']) if (team['w'] + team['l']) > 0 else 0.5
                })
    return records

# ============================================================
# 1. 資料蒐集
# ============================================================
print("📡 正在從 MLB Stats API 抓取數據...")
year = 2025

# 先試著抓 2025，如果沒數據就降級到 2024
try:
    bat = get_team_batting_stats(year)
    pitch = get_team_pitching_stats(year)
    if bat.empty and pitch.empty:
        raise ValueError("2025 無數據")
except:
    print("⚠️ 2025 數據尚不可用，改用 2024 年數據")
    year = 2024
    bat = get_team_batting_stats(year)
    pitch = get_team_pitching_stats(year)

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
# 保留實際存在的欄位
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
