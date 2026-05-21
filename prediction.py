import os
import sys
import json
import traceback
import csv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import UnifiedSportsModel

try:
    from scripts.elo import MLBElosystem
except:
    MLBElosystem = None
try:
    from scripts.monte_carlo import MonteCarloSimulator
except:
    MonteCarloSimulator = None

# 尝试加载模型
model = None
try:
    import joblib
    model = joblib.load("data/calibrator.pkl")
    print("已加载训练好的模型（XGBoost + 校准）")
except:
    print("未找到训练模型，将使用手工集成")

# 加载休息天数缓存
LAST_GAME_FILE = "data/team_last_game.json"
if os.path.exists(LAST_GAME_FILE):
    with open(LAST_GAME_FILE, 'r') as f:
        last_game_dict = json.load(f)
else:
    last_game_dict = {}

# 球队名到 MLB Stats API team_id 的映射（用于牛棚数据）
TEAM_ID_MAP = {
    "Braves": 144, "Orioles": 110, "Red Sox": 111,
    "Cubs": 112, "White Sox": 145, "Reds": 113,
    "Guardians": 114, "Rockies": 115, "Tigers": 116,
    "Astros": 117, "Royals": 118, "Angels": 108,
    "Dodgers": 119, "Marlins": 146, "Brewers": 158,
    "Twins": 142, "Mets": 121, "Yankees": 147,
    "Athletics": 133, "Phillies": 143, "Pirates": 134,
    "Padres": 135, "Giants": 137, "Mariners": 136,
    "Cardinals": 138, "Rays": 139, "Rangers": 140,
    "Blue Jays": 141, "Nationals": 120, "D-backs": 109
}

def implied_prob(odds):
    if odds is None or odds <= 1:
        return None
    return 1 / (odds * 1.05)

def kelly_criterion(win_prob, odds, fraction=0.25):
    if win_prob is None or odds is None or odds <= 1:
        return 0
    b = odds - 1
    f = win_prob - (1 - win_prob) / b
    return max(0, f * fraction)

def generate_predictions(elo_system=None):
    print("开始抓取数据...")
    m = UnifiedSportsModel()
    data = m.gather_all_data()
    date_str = datetime.now().strftime('%Y-%m-%d')
    errors = data.get('errors', [])

    if elo_system is None:
        if MLBElosystem:
            elo_system = MLBElosystem()
            print("ELO 系统已加载")
        else:
            print("ELO 模块未安装，将使用基础预测")
            elo_system = None

    # 球队战力
    teams_df = pd.DataFrame(data.get('sportsipy_teams', []))
    if teams_df.empty:
        teams_df = pd.DataFrame(columns=['name', 'wins', 'losses', 'win_pct'])
    if 'wins' in teams_df.columns and 'losses' in teams_df.columns:
        teams_df['win_pct'] = pd.to_numeric(teams_df['wins'], errors='coerce') / (
            pd.to_numeric(teams_df['wins'], errors='coerce') + pd.to_numeric(teams_df['losses'], errors='coerce')
        )
    else:
        teams_df['win_pct'] = 0.5
    teams_df['win_pct'] = teams_df['win_pct'].fillna(0.5)

    # 赔率字典
    odds_df = pd.DataFrame(data.get('odds_data', []))
    odds_dict = {}
    for _, row in odds_df.iterrows():
        key = (row.get('home_team'), row.get('away_team'))
        if key not in odds_dict:
            odds_dict[key] = []
        odds_dict[key].append(row.get('odds'))

    # 赛程
    schedule_df = pd.DataFrame(data.get('mlb_statsapi', []))
    print(f"当日比赛数量: {len(schedule_df)}")

    # 投手数据
    pitchers_df = pd.DataFrame(data.get('pitchers', []))
    pitcher_dict = {}
    if not pitchers_df.empty:
        for _, row in pitchers_df.iterrows():
            pitcher_dict[row['game_id']] = row

    # 牛棚数据
    bullpen_df = pd.DataFrame(data.get('bullpen', []))
    bullpen_dict = {}
    if not bullpen_df.empty:
        for _, row in bullpen_df.iterrows():
            bullpen_dict[row['team_id']] = row

    # 队名映射
    team_name_map = {
        "Cleveland Guardians": "Guardians",
        "Detroit Tigers": "Tigers",
        "Tampa Bay Rays": "Rays",
        "Baltimore Orioles": "Orioles",
        "Philadelphia Phillies": "Phillies",
        "Cincinnati Reds": "Reds",
        "Miami Marlins": "Marlins",
        "Atlanta Braves": "Braves",
        "Washington Nationals": "Nationals",
        "New York Mets": "Mets",
        "New York Yankees": "Yankees",
        "Toronto Blue Jays": "Blue Jays",
        "Kansas City Royals": "Royals",
        "Boston Red Sox": "Red Sox",
        "Minnesota Twins": "Twins",
        "Houston Astros": "Astros",
        "Chicago Cubs": "Cubs",
        "Milwaukee Brewers": "Brewers",
        "Colorado Rockies": "Rockies",
        "Texas Rangers": "Rangers",
        "Los Angeles Angels": "Angels",
        "Oakland Athletics": "Athletics",
        "Athletics": "Athletics",
        "San Diego Padres": "Padres",
        "Los Angeles Dodgers": "Dodgers",
        "Arizona Diamondbacks": "D-backs",
        "San Francisco Giants": "Giants",
        "Seattle Mariners": "Mariners",
        "Chicago White Sox": "White Sox",
        "Pittsburgh Pirates": "Pirates",
        "St. Louis Cardinals": "Cardinals",
    }

    # 统计历史比赛数量，用于动态融合
    HIST_FILE = "data/historical_predictions.csv"
    historical_count = 0
    if os.path.exists(HIST_FILE):
        try:
            hist_df = pd.read_csv(HIST_FILE)
            # 只计算已经有结果的比赛
            if 'home_win' in hist_df.columns:
                hist_df = hist_df[hist_df['home_win'].notna()]
                historical_count = len(hist_df)
        except:
            pass

    predictions = []
    for _, game in schedule_df.iterrows():
        home = game.get('home_team', '')
        away = game.get('away_team', '')
        home = team_name_map.get(home, home)
        away = team_name_map.get(away, away)
        if not home or not away or home == 'Unknown' or away == 'Unknown':
            continue

        # 基础胜率
        home_pct = teams_df[teams_df['name'] == home]['win_pct'].values
        away_pct = teams_df[teams_df['name'] == away]['win_pct'].values
        if len(home_pct) == 0 or len(away_pct) == 0:
            continue
        home_pct, away_pct = home_pct[0], away_pct[0]

        # ELO 差值
        elo_diff = 0.0
        if elo_system:
            elo_diff = elo_system.elos.get(home, 1500) - elo_system.elos.get(away, 1500) + elo_system.home_adv

        # 赔率
        avg_odds = odds_dict.get((home, away), [])
        home_odds = np.mean(avg_odds) if avg_odds else None
        market_prob = implied_prob(home_odds) if home_odds else 0.5

        # 投手 ERA 差值
        pitcher_data = pitcher_dict.get(game.get('game_id'))
        sp_era_diff = 0.0
        home_era = 4.5
        away_era = 4.5
        if pitcher_data is not None:
            home_era = float(pitcher_data.get('home_era') or 4.5)
            away_era = float(pitcher_data.get('away_era') or 4.5)
            sp_era_diff = home_era - away_era

        # 投手 FIP 差值
        sp_fip_diff = 0.0
        if pitcher_data is not None:
            home_fip = float(pitcher_data.get('home_fip') or 4.0)
            away_fip = float(pitcher_data.get('away_fip') or 4.0)
            sp_fip_diff = home_fip - away_fip

        # 休息天数差异
        try:
            today = datetime.strptime(date_str, "%Y-%m-%d")
            home_rest = 2
            away_rest = 2
            if home in last_game_dict:
                last_home = datetime.strptime(last_game_dict[home], "%Y-%m-%d")
                home_rest = max(0, (today - last_home).days - 1)
            if away in last_game_dict:
                last_away = datetime.strptime(last_game_dict[away], "%Y-%m-%d")
                away_rest = max(0, (today - last_away).days - 1)
            rest_diff = home_rest - away_rest
        except:
            rest_diff = 0

        # 牛棚局数差值（疲劳度）
        bullpen_ip_diff = 0.0
        home_id = TEAM_ID_MAP.get(home)
        away_id = TEAM_ID_MAP.get(away)
        if home_id and away_id and home_id in bullpen_dict and away_id in bullpen_dict:
            home_ip = bullpen_dict[home_id].get('bullpen_innings')
            away_ip = bullpen_dict[away_id].get('bullpen_innings')
            try:
                home_val = float(home_ip) if home_ip is not None else 0.0
                away_val = float(away_ip) if away_ip is not None else 0.0
                bullpen_ip_diff = home_val - away_val
            except (TypeError, ValueError):
                pass

        # 球场因子
        from scripts.park_factors import get_park_factor
        park_factor = get_park_factor(game.get('venue', ''))

        # 特征向量
        features = {
            'elo_diff': round(elo_diff, 3),
            'market_prob': round(market_prob, 3) if market_prob else 0.5,
            'sp_era_diff': round(sp_era_diff, 3),
            'sp_fip_diff': round(sp_fip_diff, 3),
            'bullpen_ip_diff': round(bullpen_ip_diff, 3),
            'rest_diff': rest_diff,
            'park_factor': park_factor,
            'home_winrate': round(home_pct, 3),
            'away_winrate': round(away_pct, 3)
        }

        # 手工集成预测（基线）
        weights = {'pct': 0.25, 'elo': 0.35, 'market': 0.40}
        if elo_system is None:
            weights['elo'] = 0
            weights['pct'] += 0.175
            weights['market'] += 0.175
        if market_prob is None or home_odds is None:
            weights['market'] = 0
            if elo_system is not None:
                weights['elo'] += 0.20
                weights['pct'] += 0.20
            else:
                weights['pct'] += 0.40

        elo_prob = 1 / (1 + 10 ** (-elo_diff / 400)) if elo_system else 0.5
        manual_pred = home_pct * weights['pct'] + elo_prob * weights['elo'] + (market_prob or 0.5) * weights['market']
        sp_adj = -0.07 * sp_era_diff
        manual_pred = min(0.95, max(0.05, manual_pred + sp_adj))

        # 机器学习预测（如果模型存在）
        ml_pred = None
        if model is not None:
            # 使用当前特征（与训练时顺序一致）
            feature_array = np.array([[
                features['elo_diff'],
                features['market_prob'],
                features['sp_era_diff'],
                features['sp_fip_diff'],
                features['bullpen_ip_diff'],
                features['rest_diff'],
                features['park_factor']
            ]])
            try:
                ml_pred = model.predict_proba(feature_array)[0, 1]
                ml_pred = min(0.95, max(0.05, ml_pred))
            except:
                ml_pred = None

        # 动态融合
        if ml_pred is not None and historical_count > 100:
            ml_weight = min(0.5, historical_count / 1000)
            pred_home = (1 - ml_weight) * manual_pred + ml_weight * ml_pred
        else:
            pred_home = manual_pred

        pred_away = 1 - pred_home

        # 凯利
        kelly_ml = 0
        kelly_ml_away = 0
        if home_odds:
            kelly_ml = kelly_criterion(pred_home, home_odds)
            kelly_ml_away = kelly_criterion(pred_away, 1 / (1 - implied_prob(home_odds))
                                            if implied_prob(home_odds) and implied_prob(home_odds) < 1 else None)

        # 蒙特卡洛模拟
        sim = None
        home_cover = away_cover = over_prob = under_prob = None
        total_mean = diff_mean = None
        ci_diff = []
        if MonteCarloSimulator:
            try:
                home_runs_col = teams_df[teams_df['name'] == home]['runs_scored'].values if 'runs_scored' in teams_df.columns else []
                away_runs_col = teams_df[teams_df['name'] == away]['runs_scored'].values if 'runs_scored' in teams_df.columns else []
                hr = float(home_runs_col[0]) if len(home_runs_col) > 0 else 4.5
                ar = float(away_runs_col[0]) if len(away_runs_col) > 0 else 4.5
                # 使用公园因子调整期望得分
                hr_adj = hr * park_factor
                ar_adj = ar * park_factor
                sim = MonteCarloSimulator(hr_adj, ar_adj, n_simulations=5000)
                sim.simulate()
                home_cover, away_cover = sim.spread_prob(-1.5)
                over_prob, under_prob, _ = sim.total_prob(8.5)
                total_mean = round(sim.results['total_runs'].mean(), 1)
                diff_mean = round(sim.results['run_diff'].mean(), 1)
                ci_diff = [round(x, 1) for x in sim.confidence_interval()]
            except:
                pass

        # 推荐
        ml_rec = "PASS"
        if kelly_ml > 0.05:
            ml_rec = f"Bet {home} ({pred_home:.1%}, {kelly_ml:.1%} Kelly)"
        elif kelly_ml_away > 0.05:
            ml_rec = f"Bet {away} ({pred_away:.1%}, {kelly_ml_away:.1%} Kelly)"

        spread_rec = "PASS"
        if home_cover is not None and home_cover > 0.55:
            spread_rec = f"Bet {home} -1.5 ({home_cover:.1%} cover)"
        elif away_cover is not None and away_cover > 0.55:
            spread_rec = f"Bet {away} +1.5 ({away_cover:.1%} cover)"

        total_rec = "PASS"
        if over_prob is not None and over_prob > 0.55:
            total_rec = f"Bet OVER 8.5 ({over_prob:.1%})"
        elif under_prob is not None and under_prob > 0.55:
            total_rec = f"Bet UNDER 8.5 ({under_prob:.1%})"

        predictions.append({
            "game_id": game.get("game_id"),
            "game_date": game.get("game_date"),
            "home_team": home,
            "away_team": away,
            "status": game.get("status"),
            "venue": game.get("venue", ""),
            "predicted_home_win_pct": round(pred_home, 3),
            "predicted_away_win_pct": round(pred_away, 3),
            "home_odds": home_odds,
            "elo_home": elo_system.elos.get(home, 1500) if elo_system else 1500,
            "elo_away": elo_system.elos.get(away, 1500) if elo_system else 1500,
            "moneyline_recommendation": ml_rec,
            "spread_recommendation": spread_rec,
            "total_recommendation": total_rec,
            "simulated_total_mean": total_mean,
            "simulated_diff_mean": diff_mean,
            "confidence_interval_diff": ci_diff,
            "over_prob": round(over_prob, 3) if over_prob is not None else None,
            "under_prob": round(under_prob, 3) if under_prob is not None else None,
            "home_cover_prob": round(home_cover, 3) if home_cover is not None else None,
            "away_cover_prob": round(away_cover, 3) if away_cover is not None else None,
            "kelly_fraction": round(kelly_ml, 4),
            "elo_diff": features['elo_diff'],
            "market_prob": features['market_prob'],
            "sp_era_diff": features['sp_era_diff'],
            "sp_fip_diff": features['sp_fip_diff'],
            "bullpen_ip_diff": features['bullpen_ip_diff'],
            "rest_diff": features['rest_diff'],
            "park_factor": features['park_factor'],
            "home_winrate": features['home_winrate'],
            "away_winrate": features['away_winrate']
        })

    # 战力排名
    power_rankings = teams_df.sort_values('win_pct', ascending=False).to_dict('records')

    output = {
        "generated_at": datetime.now().isoformat(),
        "power_rankings": power_rankings,
        "elo_ratings": {k: round(v, 1) for k, v in elo_system.elos.items()} if elo_system else {},
        "today_predictions": predictions,
        "bet_summary": {
            "moneyline_bets": [p for p in predictions if p['moneyline_recommendation'] != 'PASS'],
            "spread_bets": [p for p in predictions if p['spread_recommendation'] != 'PASS'],
            "total_bets": [p for p in predictions if p['total_recommendation'] != 'PASS']
        },
        "errors": errors
    }

    # 保存历史预测记录
    HISTORY_FILE = "data/historical_predictions.csv"
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "game_id", "game_date", "home_team", "away_team",
                "pred_home_win", "home_odds", "elo_home", "elo_away",
                "ml_rec", "spread_rec", "total_rec",
                "kelly_fraction", "home_win",
                "elo_diff", "market_prob", "sp_era_diff", "sp_fip_diff", "bullpen_ip_diff", "rest_diff", "park_factor"
            ])
        for p in predictions:
            writer.writerow([
                p.get("game_id", ""),
                p.get("game_date", ""),
                p.get("home_team", ""),
                p.get("away_team", ""),
                p.get("predicted_home_win_pct", ""),
                p.get("home_odds", ""),
                p.get("elo_home", ""),
                p.get("elo_away", ""),
                p.get("moneyline_recommendation", ""),
                p.get("spread_recommendation", ""),
                p.get("total_recommendation", ""),
                p.get("kelly_fraction", ""),
                "",
                p.get("elo_diff", ""),
                p.get("market_prob", ""),
                p.get("sp_era_diff", ""),
                p.get("sp_fip_diff", ""),
                p.get("bullpen_ip_diff", ""),
                p.get("rest_diff", ""),
                p.get("park_factor", "")
            ])
    print(f"历史预测已追加至 {HISTORY_FILE}")

    # 保存预测报告
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
