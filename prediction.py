"""
MLB 價值投注預測模型
運行方式：python prediction.py
輸出：report/prediction.json
"""
import os, json
from datetime import datetime
import pandas as pd
from model import UnifiedSportsModel

def implied_probability(odds):
    """賠率轉隱含概率（扣除 5% 抽水）"""
    return 1 / odds * 0.95

def calculate_value(home_prob, away_prob, home_odds, away_odds):
    """計算兩隊的價值分數"""
    home_value = home_prob - implied_probability(home_odds) if home_odds else 0
    away_value = away_prob - implied_probability(away_odds) if away_odds else 0
    return home_value, away_value

def generate_predictions():
    m = UnifiedSportsModel()
    data = m.gather_all_data()

    # 1. 從球隊戰績獲取勝率（用最近30場簡單模擬，這裡直接用賽季勝率）
    teams_df = pd.DataFrame(data.get('sportsipy_teams', []))
    if teams_df.empty:
        teams_df = pd.DataFrame(columns=['name', 'wins', 'losses'])
    teams_df['win_pct'] = teams_df['wins'].astype(float) / (teams_df['wins'].astype(float) + teams_df['losses'].astype(float))
    teams_df['win_pct'] = teams_df['win_pct'].fillna(0.5)

    # 2. 從賠率數據獲取即時盤口
    odds_df = pd.DataFrame(data.get('odds_data', []))
    # 取每場比賽的平均賠率
    if not odds_df.empty:
        odds_avg = odds_df.groupby(['home_team', 'away_team'])['odds'].mean().reset_index()
    else:
        odds_avg = pd.DataFrame()

    # 3. 從賽程獲取今日對戰
    schedule_df = pd.DataFrame(data.get('mlb_statsapi', []))
    # 過濾掉非預覽狀態的比賽
    schedule_df = schedule_df[schedule_df['status'] == 'Preview']

    predictions = []
    for _, game in schedule_df.iterrows():
        home = game.get('home', '')
        away = game.get('away', '')
        # 查找兩隊勝率
        home_pct = teams_df[teams_df['name'] == home]['win_pct'].values
        away_pct = teams_df[teams_df['name'] == away]['win_pct'].values
        if len(home_pct) == 0 or len(away_pct) == 0:
            continue
        home_pct = home_pct[0]
        away_pct = away_pct[0]

        # 簡單預測：主場優勢 +0.04，然後標準化
        raw_home = home_pct * 0.6 + 0.04
        raw_away = away_pct * 0.4
        total = raw_home + raw_away
        pred_home = raw_home / total
        pred_away = raw_away / total

        # 查找賠率
        home_odds = None
        away_odds = None
        if not odds_avg.empty:
            match_odds = odds_avg[(odds_avg['home_team'] == home) & (odds_avg['away_team'] == away)]
            if not match_odds.empty:
                home_odds = match_odds.iloc[0]['odds']
                # 客隊賠率需要另一條記錄，這裡簡化用同一平均（實務上應分開）
                # 假設 h2h 市場有兩個 outcome，此處用同一值演示，可自行擴展
                away_odds = home_odds  # 簡化，真實應從 outcomes 取得

        home_val, away_val = calculate_value(pred_home, pred_away, home_odds, away_odds)

        recommendation = None
        if home_val > 0.05:
            recommendation = f"Bet {home} (Value: {home_val:.2%})"
        elif away_val > 0.05:
            recommendation = f"Bet {away} (Value: {away_val:.2%})"

        predictions.append({
            "home_team": home,
            "away_team": away,
            "predicted_home_win_pct": round(pred_home, 3),
            "predicted_away_win_pct": round(pred_away, 3),
            "home_odds": home_odds,
            "away_odds": away_odds,
            "home_value": round(home_val, 4),
            "away_value": round(away_val, 4),
            "recommendation": recommendation
        })

    # 加入球隊戰力總排名
    power_rankings = teams_df.sort_values('win_pct', ascending=False)[['name', 'wins', 'losses', 'win_pct']].to_dict('records')

    output = {
        "generated_at": datetime.now().isoformat(),
        "power_rankings": power_rankings,
        "today_predictions": predictions
    }

    os.makedirs('report', exist_ok=True)
    with open('report/prediction.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("Prediction saved to report/prediction.json")

if __name__ == '__main__':
    generate_predictions()
