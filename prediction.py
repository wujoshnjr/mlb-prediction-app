"""
MLB 价值投注预测模型（混合架构版）
包含 ELO + 特征工程 + 凯利准则
"""
import os, json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from model import UnifiedSportsModel
from scripts.elo import MLBElosystem

def implied_probability(odds):
    """赔率转隐含概率，扣除市场抽水"""
    if odds is None or odds <= 1:
        return None
    overround = 1.05  # 5% 抽水
    return 1 / (odds * overround)

def kelly_criterion(win_prob, odds, fraction=0.25):
    """凯利准则：计算最优下注比例"""
    if win_prob is None or odds is None or odds <= 1:
        return 0
    b = odds - 1
    f = win_prob - (1 - win_prob) / b
    return max(0, f * fraction)

def calculate_matchup_score(home_stats, away_stats, stat_key, home_pitcher, away_pitcher):
    """计算主客队特定统计指标的标准化差值"""
    # ... (此函数逻辑从略，核心是利用统计差值进行标准化)

def generate_predictions(elo_system=None):
    m = UnifiedSportsModel()
    data = m.gather_all_data()
    date_str = datetime.now().strftime('%Y-%m-%d')

    # 初始化 ELO（首次运行从1500开始）
    if elo_system is None:
        elo_system = MLBElosystem()

    # 1. 获取球队战力
    teams_df = pd.DataFrame(data.get('sportsipy_teams', []))
    if teams_df.empty:
        teams_df = pd.DataFrame(columns=['name', 'wins', 'losses'])
    if 'wins' in teams_df.columns and 'losses' in teams_df.columns:
        teams_df['win_pct'] = teams_df['wins'].astype(float) / (teams_df['wins'].astype(float) + teams_df['losses'].astype(float))
    else:
        teams_df['win_pct'] = 0.5

    # 2. 构建赔率字典
    odds_df = pd.DataFrame(data.get('odds_data', []))
    odds_dict = {}
    for _, row in odds_df.iterrows():
        key = (row.get('home_team'), row.get('away_team'))
        if key not in odds_dict:
            odds_dict[key] = []
        odds_dict[key].append(row.get('odds'))

    # 3. 处理赛程
    schedule_df = pd.DataFrame(data.get('mlb_statsapi', []))
    schedule_df = schedule_df[schedule_df['status'] == 'Preview']

    predictions = []
    for _, game in schedule_df.iterrows():
        home = game.get('home', '')
        away = game.get('away', '')
        if not home or not away or home == 'Unknown' or away == 'Unknown':
            continue

        # 获取两队胜率
        home_pct = teams_df[teams_df['name'] == home]['win_pct'].values
        away_pct = teams_df[teams_df['name'] == away]['win_pct'].values
        if len(home_pct) == 0 or len(away_pct) == 0:
            continue
        home_pct, away_pct = home_pct[0], away_pct[0]

        # 获取 ELO 预测
        elo_diff = elo_system.elos.get(home, 1500) - elo_system.elos.get(away, 1500) + elo_system.home_adv
        elo_win_prob = 1 / (1 + 10 ** (-elo_diff / 400))

        # 获取赔率
        avg_odds = odds_dict.get((home, away), [])
        home_odds = np.mean(avg_odds) if avg_odds else None
        away_odds = 1 / (1 - implied_probability(home_odds)) if home_odds else None

        # 模型集成预测
        pred_home = (home_pct * 0.25 + elo_win_prob * 0.35 + implied_probability(home_odds) * 0.40) if home_odds else (home_pct * 0.4 + elo_win_prob * 0.6)
        pred_away = 1 - pred_home

        # 凯利准则
        kelly_fraction = kelly_criterion(pred_home, home_odds) if home_odds else 0
        recommendation = None
        if kelly_fraction > 0.05:
            recommendation = f"Bet {home} ({pred_home:.1%} value, {kelly_fraction:.1%} Kelly)"
        elif kelly_criterion(pred_away, away_odds) > 0.05:
            recommendation = f"Bet {away} ({pred_away:.1%} value, {kelly_criterion(pred_away, away_odds):.1%} Kelly)"

        predictions.append({
            "home_team": home, "away_team": away,
            "predicted_home_win_pct": round(pred_home, 3),
            "predicted_away_win_pct": round(pred_away, 3),
            "home_odds": home_odds, "away_odds": away_odds,
            "kelly_fraction": round(kelly_fraction, 4),
            "elo_home": elo_system.elos.get(home, 1500),
            "elo_away": elo_system.elos.get(away, 1500),
            "recommendation": recommendation
        })

    # 球队战力排名
    if 'win_pct' in teams_df.columns:
        power_rankings = teams_df.sort_values('win_pct', ascending=False)[['name', 'wins', 'losses', 'win_pct']].to_dict('records')
    else:
        power_rankings = []

    output = {
        "generated_at": datetime.now().isoformat(),
        "power_rankings": power_rankings,
        "today_predictions": predictions,
        "elo_ratings": {k: round(v, 1) for k, v in elo_system.elos.items()}
    }

    os.makedirs('report', exist_ok=True)
    with open('report/prediction.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("Prediction saved to report/prediction.json")
    return output

if __name__ == '__main__':
    generate_predictions()
