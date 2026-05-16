"""
完整投注決策引擎
支援：勝負盤、讓分盤、大小分盤 + 蒙地卡羅模擬
"""
import os, json
from datetime import datetime
import pandas as pd
import numpy as np
from model import UnifiedSportsModel
from scripts.elo import MLBElosystem
from scripts.monte_carlo import MonteCarloSimulator

def implied_prob(odds):
    """賠率轉隱含概率（扣除抽水）"""
    if odds is None or odds <= 1:
        return None
    return 1 / (odds * 1.05)  # 5% overround

def kelly_criterion(win_prob, odds, fraction=0.25):
    """半凱利準則"""
    if win_prob is None or odds is None or odds <= 1:
        return 0
    b = odds - 1
    f = win_prob - (1 - win_prob) / b
    return max(0, f * fraction)

def generate_predictions(elo_system=None):
    m = UnifiedSportsModel()
    data = m.gather_all_data()
    date_str = datetime.now().strftime('%Y-%m-%d')

    if elo_system is None:
        elo_system = MLBElosystem()

    # ----- 球隊戰力 -----
    teams_df = pd.DataFrame(data.get('sportsipy_teams', []))
    if teams_df.empty:
        teams_df = pd.DataFrame(columns=['name', 'wins', 'losses'])
    if 'wins' in teams_df.columns and 'losses' in teams_df.columns:
        teams_df['win_pct'] = teams_df['wins'].astype(float) / (teams_df['wins'].astype(float) + teams_df['losses'].astype(float))
    else:
        teams_df['win_pct'] = 0.5
    teams_df['runs_scored'] = teams_df.get('runs_scored', pd.Series([4.5] * len(teams_df)))
    teams_df['runs_allowed'] = teams_df.get('runs_allowed', pd.Series([4.5] * len(teams_df)))

    # ----- 賠率 -----
    odds_df = pd.DataFrame(data.get('odds_data', []))
    odds_dict = {}
    for _, row in odds_df.iterrows():
        key = (row.get('home_team'), row.get('away_team'))
        if key not in odds_dict:
            odds_dict[key] = []
        odds_dict[key].append(row.get('odds'))

    # ----- 賽程 -----
    schedule_df = pd.DataFrame(data.get('mlb_statsapi', []))
    schedule_df = schedule_df[schedule_df['status'] == 'Preview']

    predictions = []
    for _, game in schedule_df.iterrows():
        home = game.get('home', '')
        away = game.get('away', '')
        if not home or not away or home == 'Unknown' or away == 'Unknown':
            continue

        # ----- 基礎勝率 -----
        home_pct = teams_df[teams_df['name'] == home]['win_pct'].values
        away_pct = teams_df[teams_df['name'] == away]['win_pct'].values
        if len(home_pct) == 0 or len(away_pct) == 0:
            continue
        home_pct, away_pct = home_pct[0], away_pct[0]

        # ----- ELO 預測 -----
        elo_diff = elo_system.elos.get(home, 1500) - elo_system.elos.get(away, 1500) + elo_system.home_adv
        elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))

        # ----- 賠率 -----
        avg_odds = odds_dict.get((home, away), [])
        home_odds = np.mean(avg_odds) if avg_odds else 2.0  # 預設

        # ----- 勝負盤 (Moneyline) -----
        market_implied = implied_prob(home_odds) or 0.5
        pred_home = home_pct * 0.25 + elo_prob * 0.35 + market_implied * 0.40
        pred_away = 1 - pred_home
        kelly_ml = kelly_criterion(pred_home, home_odds)
        kelly_ml_away = kelly_criterion(pred_away, 1 / (1 - market_implied) if market_implied < 1 else 2.0)

        # 勝負盤推薦
        ml_rec = "PASS"
        if kelly_ml > 0.05:
            ml_rec = f"Bet {home} ({pred_home:.1%} value, {kelly_ml:.1%} Kelly)"
        elif kelly_ml_away > 0.05:
            ml_rec = f"Bet {away} ({pred_away:.1%} value, {kelly_ml_away:.1%} Kelly)"

        # ----- 蒙地卡羅模擬 -----
        home_runs_avg = float(teams_df[teams_df['name'] == home]['runs_scored'].values[0]) if len(teams_df[teams_df['name'] == home]) > 0 else 4.5
        away_runs_avg = float(teams_df[teams_df['name'] == away]['runs_scored'].values[0]) if len(teams_df[teams_df['name'] == away]) > 0 else 4.5
        sim = MonteCarloSimulator(home_runs_avg, away_runs_avg, n_simulations=5000)
        sim.simulate()

        # ----- 讓分盤 (Spread) -----
        spread = -1.5  # 標準讓分線
        home_cover, away_cover = sim.spread_prob(spread)
        spread_rec = "PASS"
        if home_cover > 0.55:
            spread_rec = f"Bet {home} -1.5 ({home_cover:.1%} cover)"
        elif away_cover > 0.55:
            spread_rec = f"Bet {away} +1.5 ({away_cover:.1%} cover)"

        # ----- 大小分盤 (Total) -----
        total_line = 8.5  # 標準大小分線
        over_prob, under_prob, push_prob = sim.total_prob(total_line)
        total_rec = "PASS"
        if over_prob > 0.55:
            total_rec = f"Bet OVER {total_line} ({over_prob:.1%} prob)"
        elif under_prob > 0.55:
            total_rec = f"Bet UNDER {total_line} ({under_prob:.1%} prob)"

        # ----- 置信區間 -----
        ci = sim.confidence_interval('run_diff', confidence=0.80)

        predictions.append({
            "home_team": home, "away_team": away,
            "predicted_home_win_pct": round(pred_home, 3),
            "predicted_away_win_pct": round(pred_away, 3),
            "home_odds": home_odds,
            "elo_home": elo_system.elos.get(home, 1500),
            "elo_away": elo_system.elos.get(away, 1500),
            # 三種玩法推薦
            "moneyline_recommendation": ml_rec,
            "spread_recommendation": spread_rec,
            "spread_line": spread,
            "total_recommendation": total_rec,
            "total_line": total_line,
            # 蒙地卡羅模擬結果
            "simulated_total_mean": round(sim.results['total_runs'].mean(), 1),
            "simulated_diff_mean": round(sim.results['run_diff'].mean(), 1),
            "confidence_interval_diff": [round(ci[0], 1), round(ci[1], 1)],
            "over_prob": round(over_prob, 3),
            "under_prob": round(under_prob, 3)
        })

    # ----- 輸出 -----
    output = {
        "generated_at": datetime.now().isoformat(),
        "power_rankings": teams_df[['name', 'wins', 'losses', 'win_pct']].sort_values('win_pct', ascending=False).to_dict('records'),
        "elo_ratings": {k: round(v, 1) for k, v in elo_system.elos.items()},
        "today_predictions": predictions,
        "bet_summary": {
            "moneyline_bets": [p for p in predictions if p['moneyline_recommendation'] != 'PASS'],
            "spread_bets": [p for p in predictions if p['spread_recommendation'] != 'PASS'],
            "total_bets": [p for p in predictions if p['total_recommendation'] != 'PASS']
        }
    }

    os.makedirs('report', exist_ok=True)
    with open('report/prediction.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    return output
