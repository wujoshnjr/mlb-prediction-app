"""
MLB 混合预测引擎 v3.0
整合 ELO、赔率、蒙特卡洛、凯利准则
支持：胜分差、让分、大小分
"""
import os, json, traceback
from datetime import datetime
import pandas as pd
import numpy as np
from model import UnifiedSportsModel

# 尝试导入 ELO 和蒙特卡洛，失败则降级运行
try:
    from scripts.elo import MLBElosystem
except:
    MLBElosystem = None
try:
    from scripts.monte_carlo import MonteCarloSimulator
except:
    MonteCarloSimulator = None

def implied_prob(odds):
    """赔率转隐含概率（扣除 5% 抽水）"""
    if odds is None or odds <= 1: return None
    return 1 / (odds * 1.05)

def kelly_criterion(win_prob, odds, fraction=0.25):
    if win_prob is None or odds is None or odds <= 1: return 0
    b = odds - 1
    f = win_prob - (1 - win_prob) / b
    return max(0, f * fraction)

def generate_predictions(elo_system=None):
    print("开始抓取数据...")
    m = UnifiedSportsModel()
    data = m.gather_all_data()
    date_str = datetime.now().strftime('%Y-%m-%d')
    errors = data.get('errors', [])

    # ----- ELO 初始化 -----
    if elo_system is None:
        if MLBElosystem:
            elo_system = MLBElosystem()
            print("ELO 系统已加载")
        else:
            print("ELO 模块未安装，将使用基础预测")
            elo_system = None

    # ----- 球队战力 -----
    teams_df = pd.DataFrame(data.get('sportsipy_teams', []))
    if teams_df.empty:
        teams_df = pd.DataFrame(columns=['name', 'wins', 'losses', 'win_pct'])
    if 'wins' in teams_df.columns and 'losses' in teams_df.columns:
        teams_df['win_pct'] = pd.to_numeric(teams_df['wins'], errors='coerce') / (pd.to_numeric(teams_df['wins'], errors='coerce') + pd.to_numeric(teams_df['losses'], errors='coerce'))
    else:
        teams_df['win_pct'] = 0.5

    # ----- 赔率字典 -----
    odds_df = pd.DataFrame(data.get('odds_data', []))
    odds_dict = {}
    for _, row in odds_df.iterrows():
        key = (row.get('home_team'), row.get('away_team'))
        if key not in odds_dict: odds_dict[key] = []
        odds_dict[key].append(row.get('odds'))

    # ----- 赛程 -----
    schedule_df = pd.DataFrame(data.get('mlb_statsapi', []))
    schedule_df = schedule_df[schedule_df['status'] == 'Preview']

    predictions = []
    for _, game in schedule_df.iterrows():
        home = game.get('home', '')
        away = game.get('away', '')
        if not home or not away or home == 'Unknown' or away == 'Unknown':
            continue

        # 基础胜率
        home_pct = teams_df[teams_df['name'] == home]['win_pct'].values
        away_pct = teams_df[teams_df['name'] == away]['win_pct'].values
        if len(home_pct) == 0 or len(away_pct) == 0: continue
        home_pct, away_pct = home_pct[0], away_pct[0]

        # ELO 预测
        elo_prob = 0.5
        if elo_system:
            elo_diff = elo_system.elos.get(home, 1500) - elo_system.elos.get(away, 1500) + elo_system.home_adv
            elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))

        # 赔率预测
        avg_odds = odds_dict.get((home, away), [])
        home_odds = np.mean(avg_odds) if avg_odds else 2.0
        market_prob = implied_prob(home_odds) if home_odds else 0.5

        # 集成预测（加权平均）
        pred_home = home_pct * 0.25 + elo_prob * 0.35 + (market_prob or 0.5) * 0.40
        pred_away = 1 - pred_home

        # 蒙特卡洛模拟（如果有模块）
        sim = None
        if MonteCarloSimulator:
            try:
                home_runs = teams_df[teams_df['name'] == home]['runs_scored'].values
                away_runs = teams_df[teams_df['name'] == away]['runs_scored'].values
                hr = float(home_runs[0]) if len(home_runs) > 0 else 4.5
                ar = float(away_runs[0]) if len(away_runs) > 0 else 4.5
                sim = MonteCarloSimulator(hr, ar, n_simulations=5000)
                sim.simulate()
            except:
                pass

        # 玩法推荐
        ml_rec, spread_rec, total_rec = "PASS", "PASS", "PASS"
        if home_odds:
            kelly_ml = kelly_criterion(pred_home, home_odds)
            kelly_away = kelly_criterion(pred_away, 1/(1-implied_prob(home_odds)) if implied_prob(home_odds) and implied_prob(home_odds)<1 else 2.0)
            if kelly_ml > 0.05:
                ml_rec = f"Bet {home} ({pred_home:.1%}, {kelly_ml:.1%} Kelly)"
            elif kelly_away > 0.05:
                ml_rec = f"Bet {away} ({pred_away:.1%}, {kelly_away:.1%} Kelly)"

        if sim:
            home_cover, away_cover = sim.spread_prob(-1.5)
            if home_cover > 0.55:
                spread_rec = f"Bet {home} -1.5 ({home_cover:.1%} cover)"
            elif away_cover > 0.55:
                spread_rec = f"Bet {away} +1.5 ({away_cover:.1%} cover)"
            over_prob, under_prob, _ = sim.total_prob(8.5)
            if over_prob > 0.55:
                total_rec = f"Bet OVER 8.5 ({over_prob:.1%})"
            elif under_prob > 0.55:
                total_rec = f"Bet UNDER 8.5 ({under_prob:.1%})"

        predictions.append({
            "home_team": home, "away_team": away,
            "predicted_home_win_pct": round(pred_home, 3),
            "predicted_away_win_pct": round(pred_away, 3),
            "home_odds": home_odds,
            "elo_home": elo_system.elos.get(home, 1500) if elo_system else 1500,
            "elo_away": elo_system.elos.get(away, 1500) if elo_system else 1500,
            "moneyline_recommendation": ml_rec,
            "spread_recommendation": spread_rec,
            "total_recommendation": total_rec,
            "simulated_total_mean": round(sim.results['total_runs'].mean(), 1) if sim else None,
            "simulated_diff_mean": round(sim.results['run_diff'].mean(), 1) if sim else None,
            "confidence_interval_diff": [round(x,1) for x in sim.confidence_interval()] if sim else [],
            "over_prob": round(over_prob,3) if sim else None,
            "under_prob": round(under_prob,3) if sim else None,
            "home_cover_prob": round(home_cover,3) if sim else None,
            "away_cover_prob": round(away_cover,3) if sim else None,
            "kelly_fraction": round(kelly_ml,4) if home_odds else 0
        })

    # 输出
    output = {
        "generated_at": datetime.now().isoformat(),
        "power_rankings": teams_df.sort_values('win_pct', ascending=False).to_dict('records'),
        "elo_ratings": {k: round(v,1) for k,v in elo_system.elos.items()} if elo_system else {},
        "today_predictions": predictions,
        "bet_summary": {
            "moneyline_bets": [p for p in predictions if p['moneyline_recommendation'] != 'PASS'],
            "spread_bets": [p for p in predictions if p['spread_recommendation'] != 'PASS'],
            "total_bets": [p for p in predictions if p['total_recommendation'] != 'PASS']
        },
        "errors": errors
    }

    os.makedirs('report', exist_ok=True)
    with open('report/prediction.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("prediction.json 已生成")
    return output

if __name__ == '__main__':
    try:
        generate_predictions()
    except Exception as e:
        print("严重错误：", e)
        traceback.print_exc()
        exit(1)
