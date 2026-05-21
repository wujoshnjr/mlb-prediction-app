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
try:
    from scripts.catcher_utils import calculate_catcher_effect
except:
    calculate_catcher_effect = None
try:
    from scripts.lag_features import calculate_lag_features
except:
    calculate_lag_features = None
try:
    from scripts.expected_value import filter_value_bets
except:
    filter_value_bets = None

# 尝试加载训练好的模型（XGBoost + 校准）
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

# 球队时区映射（用于旅行疲劳计算）
TEAM_TIMEZONES = {
    "Braves": "Eastern", "Orioles": "Eastern", "Red Sox": "Eastern",
    "Cubs": "Central", "White Sox": "Central", "Reds": "Eastern",
    "Guardians": "Eastern", "Rockies": "Mountain", "Tigers": "Eastern",
    "Astros": "Central", "Royals": "Central", "Angels": "Pacific",
    "Dodgers": "Pacific", "Marlins": "Eastern", "Brewers": "Central",
    "Twins": "Central", "Mets": "Eastern", "Yankees": "Eastern",
    "Athletics": "Pacific", "Phillies": "Eastern", "Pirates": "Eastern",
    "Padres": "Pacific", "Giants": "Pacific", "Mariners": "Pacific",
    "Cardinals": "Central", "Rays": "Eastern", "Rangers": "Central",
    "Blue Jays": "Eastern", "Nationals": "Eastern", "D-backs": "Mountain"
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

    # 确保有 runs_scored 和 runs_allowed 列
    if 'runs_scored' not in teams_df.columns:
        teams_df['runs_scored'] = 400
    if 'runs_allowed' not in teams_df.columns:
        teams_df['runs_allowed'] = 400

    # ========== 赔率字典：只取主队赔率 ==========
    odds_df = pd.DataFrame(data.get('odds_data', []))
    odds_dict = {}
    if not odds_df.empty:
        home_odds_df = odds_df[odds_df['team'] == 'home']
        for _, row in home_odds_df.iterrows():
            key = (row.get('home_team'), row.get('away_team'))
            odds_val = row.get('odds')
            if odds_val is not None and odds_val > 1:
                if key not in odds_dict:
                    odds_dict[key] = []
                odds_dict[key].append(odds_val)

    schedule_df = pd.DataFrame(data.get('mlb_statsapi', []))
    print(f"当日比赛数量: {len(schedule_df)}")

    pitchers_df = pd.DataFrame(data.get('pitchers', []))
    pitcher_dict = {}
    if not pitchers_df.empty:
        for _, row in pitchers_df.iterrows():
            pitcher_dict[row['game_id']] = row

    bullpen_df = pd.DataFrame(data.get('bullpen', []))
    bullpen_dict = {}
    if not bullpen_df.empty:
        for _, row in bullpen_df.iterrows():
            bullpen_dict[row['team_id']] = row

    platoon_df = pd.DataFrame(data.get('platoon', []))
    platoon_dict = {}
    if not platoon_df.empty:
        for _, row in platoon_df.iterrows():
            team = row['team_name']
            split = row['split']
            if team not in platoon_dict:
                platoon_dict[team] = {}
            platoon_dict[team][split] = {'ops': float(row.get('ops', 0.700)) if row.get('ops') else 0.700}

    # Statcast 团队聚合
    savant_df = pd.DataFrame(data.get('savant_statcast', []))
    statcast_team_stats = {}
    if not savant_df.empty and 'launch_speed' in savant_df.columns:
        required_cols = ['home_team', 'away_team', 'inning_topbot', 'launch_speed', 'barrel', 'hard_hit', 'expected_woba']
        if all(c in savant_df.columns for c in required_cols):
            home_bat = savant_df[savant_df['inning_topbot'] == 'Bot'].copy()
            away_bat = savant_df[savant_df['inning_topbot'] == 'Top'].copy()
            for df_side, side_label in [(home_bat, 'home'), (away_bat, 'away')]:
                team_col = 'home_team' if side_label == 'home' else 'away_team'
                grouped = df_side.groupby(team_col).agg(
                    avg_launch_speed=('launch_speed', 'mean'),
                    barrel_rate=('barrel', lambda x: x.astype(float).eq(1).mean()),
                    hard_hit_rate=('hard_hit', lambda x: x.astype(float).eq(1).mean()),
                    avg_expected_woba=('expected_woba', 'mean')
                ).reset_index()
                grouped.rename(columns={team_col: 'team_name'}, inplace=True)
                statcast_team_stats[side_label] = grouped

    weather_df = pd.DataFrame(data.get('openmeteo_weather', []))
    avg_wind_speed = weather_df['wind_speed'].mean() if not weather_df.empty else 0
    avg_wind_dir = weather_df['wind_direction'].mean() if not weather_df.empty else 0

    # 加载历史比赛数据（用于滞后特征）
    historical_df = None
    hist_dir = "data/historical"
    if os.path.exists(hist_dir):
        hist_files = [os.path.join(hist_dir, f) for f in os.listdir(hist_dir) if f.endswith(".parquet")]
        if hist_files:
            try:
                historical_df = pd.concat([pd.read_parquet(f) for f in hist_files], ignore_index=True)
            except:
                pass

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

    HIST_FILE = "data/historical_predictions.csv"
    historical_count = 0
    if os.path.exists(HIST_FILE):
        try:
            hist_df = pd.read_csv(HIST_FILE)
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

        # 赔率（已修复，只取主队赔率）
        avg_odds = odds_dict.get((home, away), [])
        home_odds = np.mean(avg_odds) if avg_odds else None
        market_prob = implied_prob(home_odds) if home_odds else 0.5

        # 投手数据
        pitcher_data = pitcher_dict.get(game.get('game_id'))
        sp_era_diff = 0.0
        sp_fip_diff = 0.0
        home_pitch_hand = "R"
        away_pitch_hand = "R"
        if pitcher_data is not None:
            home_era = float(pitcher_data.get('home_era') or 4.5)
            away_era = float(pitcher_data.get('away_era') or 4.5)
            sp_era_diff = home_era - away_era
            home_fip = float(pitcher_data.get('home_fip') or 4.0)
            away_fip = float(pitcher_data.get('away_fip') or 4.0)
            sp_fip_diff = home_fip - away_fip
            home_pitch_hand = pitcher_data.get('home_pitch_hand', 'R')
            away_pitch_hand = pitcher_data.get('away_pitch_hand', 'R')

        # 休息天数 + 旅行疲劳
        rest_diff = 0
        timezone_diff = 0
        is_day_game = game.get('is_day_game', 0)
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

            if home in TEAM_TIMEZONES and away in TEAM_TIMEZONES:
                tz_map = {"Eastern": 0, "Central": -1, "Mountain": -2, "Pacific": -3}
                home_tz = tz_map.get(TEAM_TIMEZONES[home], 0)
                away_tz = tz_map.get(TEAM_TIMEZONES[away], 0)
                timezone_diff = away_tz - home_tz
        except:
            pass

        # 牛棚
        bullpen_ip_diff = 0.0
        home_back2back = 0
        away_back2back = 0
        home_id = TEAM_ID_MAP.get(home)
        away_id = TEAM_ID_MAP.get(away)
        if home_id and away_id and home_id in bullpen_dict and away_id in bullpen_dict:
            home_ip = bullpen_dict[home_id].get('bullpen_innings')
            away_ip = bullpen_dict[away_id].get('bullpen_innings')
            try:
                home_val = float(home_ip) if home_ip is not None else 0.0
                away_val = float(away_ip) if away_ip is not None else 0.0
                bullpen_ip_diff = home_val - away_val
            except:
                pass
            home_back2back = int(bullpen_dict[home_id].get('back_to_back', 0))
            away_back2back = int(bullpen_dict[away_id].get('back_to_back', 0))

        # 球场因子
        from scripts.park_factors import get_park_factor
        park_factor = get_park_factor(game.get('venue', ''))

        # Platoon
        platoon_ops_diff = 0.0
        if not platoon_df.empty:
            home_split = "vsLhp" if home_pitch_hand == "L" else "vsRhp"
            away_split = "vsLhp" if away_pitch_hand == "L" else "vsRhp"
            home_platoon = platoon_dict.get(home, {}).get(home_split, {})
            away_platoon = platoon_dict.get(away, {}).get(away_split, {})
            if home_platoon and away_platoon:
                platoon_ops_diff = home_platoon['ops'] - away_platoon['ops']

        # 捕手
        catcher_era_diff = 0.0
        cs_diff = 0.0
        if calculate_catcher_effect:
            home_catcher_id = game.get('home_catcher_id')
            away_catcher_id = game.get('away_catcher_id')
            if home_catcher_id and away_catcher_id:
                catcher_era_diff, cs_diff = calculate_catcher_effect(home_catcher_id, away_catcher_id, 2026)

        # Statcast
        statcast_launch_speed_diff = 0.0
        statcast_barrel_diff = 0.0
        statcast_hard_hit_diff = 0.0
        statcast_woba_diff = 0.0
        if statcast_team_stats:
            home_hit = statcast_team_stats.get('home')
            away_hit = statcast_team_stats.get('away')
            if home_hit is not None and away_hit is not None:
                home_row = home_hit[home_hit['team_name'] == home]
                away_row = away_hit[away_hit['team_name'] == away]
                if not home_row.empty and not away_row.empty:
                    statcast_launch_speed_diff = home_row.iloc[0]['avg_launch_speed'] - away_row.iloc[0]['avg_launch_speed']
                    statcast_barrel_diff = home_row.iloc[0]['barrel_rate'] - away_row.iloc[0]['barrel_rate']
                    statcast_hard_hit_diff = home_row.iloc[0]['hard_hit_rate'] - away_row.iloc[0]['hard_hit_rate']
                    statcast_woba_diff = home_row.iloc[0]['avg_expected_woba'] - away_row.iloc[0]['avg_expected_woba']

        # 天气
        wind_effect = 0.0
        if avg_wind_speed > 10:
            wind_effect = 0.02 * avg_wind_speed * np.sin(np.radians(avg_wind_dir))

        # ========== Pythagorean Record ==========
        home_runs_scored = float(teams_df[teams_df['name'] == home]['runs_scored'].values[0])
        home_runs_allowed = float(teams_df[teams_df['name'] == home]['runs_allowed'].values[0])
        away_runs_scored = float(teams_df[teams_df['name'] == away]['runs_scored'].values[0])
        away_runs_allowed = float(teams_df[teams_df['name'] == away]['runs_allowed'].values[0])

        home_pythag = home_runs_scored**1.83 / (home_runs_scored**1.83 + home_runs_allowed**1.83) if (home_runs_scored + home_runs_allowed) > 0 else 0.5
        away_pythag = away_runs_scored**1.83 / (away_runs_scored**1.83 + away_runs_allowed**1.83) if (away_runs_scored + away_runs_allowed) > 0 else 0.5
        pythag_diff = home_pythag - away_pythag

        # ========== Log5 概率 ==========
        log5_home = (home_pct - home_pct * away_pct) / (home_pct + away_pct - 2 * home_pct * away_pct) if (home_pct + away_pct) > 0 else 0.5

        # ========== 滞后特征（最近30天胜率与得分差值）==========
        lag30_winrate_diff = 0.0
        lag30_runs_diff = 0.0
        if calculate_lag_features and historical_df is not None:
            try:
                home_winrate_30, away_winrate_30, home_runs_30, away_runs_30 = calculate_lag_features(
                    home, away, historical_df, date_str, days=30
                )
                lag30_winrate_diff = home_winrate_30 - away_winrate_30
                lag30_runs_diff = home_runs_30 - away_runs_30
            except:
                pass

        # 特征向量
        features = {
            'elo_diff': round(elo_diff, 3),
            'market_prob': round(market_prob, 3) if market_prob else 0.5,
            'sp_era_diff': round(sp_era_diff, 3),
            'sp_fip_diff': round(sp_fip_diff, 3),
            'bullpen_ip_diff': round(bullpen_ip_diff, 3),
            'rest_diff': rest_diff,
            'park_factor': park_factor,
            'platoon_ops_diff': round(platoon_ops_diff, 3),
            'statcast_launch_speed_diff': round(statcast_launch_speed_diff, 3),
            'statcast_barrel_diff': round(statcast_barrel_diff, 3),
            'statcast_hard_hit_diff': round(statcast_hard_hit_diff, 3),
            'statcast_woba_diff': round(statcast_woba_diff, 3),
            'timezone_diff': timezone_diff,
            'is_day_game': is_day_game,
            'home_back2back': home_back2back,
            'away_back2back': away_back2back,
            'catcher_era_diff': round(catcher_era_diff, 3),
            'cs_diff': round(cs_diff, 3),
            'wind_effect': round(wind_effect, 4),
            'pythag_diff': round(pythag_diff, 3),
            'log5_prob': round(log5_home, 3),
            'lag30_winrate_diff': round(lag30_winrate_diff, 3),
            'lag30_runs_diff': round(lag30_runs_diff, 3),
            'home_winrate': round(home_pct, 3),
            'away_winrate': round(away_pct, 3)
        }

        # ---------- 手工集成预测 ----------
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

        # ---------- 机器学习预测 ----------
        ml_pred = None
        if model is not None:
            feature_array = np.array([[
                features['elo_diff'],
                features['market_prob'],
                features['sp_era_diff'],
                features['sp_fip_diff'],
                features['bullpen_ip_diff'],
                features['rest_diff'],
                features['park_factor'],
                features['platoon_ops_diff'],
                features['statcast_launch_speed_diff'],
                features['statcast_barrel_diff'],
                features['statcast_hard_hit_diff'],
                features['statcast_woba_diff'],
                features['timezone_diff'],
                features['is_day_game'],
                features['home_back2back'],
                features['away_back2back'],
                features['catcher_era_diff'],
                features['cs_diff'],
                features['wind_effect'],
                features['pythag_diff'],
                features['log5_prob'],
                features['lag30_winrate_diff'],
                features['lag30_runs_diff']
            ]])
            try:
                ml_pred = model.predict_proba(feature_array)[0, 1]
                ml_pred = min(0.95, max(0.05, ml_pred))
            except:
                ml_pred = None

        # ---------- 动态融合 ----------
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
                hr_adj = hr * park_factor * (1 + wind_effect)
                ar_adj = ar * park_factor * (1 - wind_effect)
                sim = MonteCarloSimulator(hr_adj, ar_adj, n_simulations=5000)
                sim.simulate()
                home_cover, away_cover = sim.spread_prob(-1.5)
                over_prob, under_prob, _ = sim.total_prob(8.5)
                total_mean = round(sim.results['total_runs'].mean(), 1)
                diff_mean = round(sim.results['run_diff'].mean(), 1)
                ci_diff = [round(x, 1) for x in sim.confidence_interval()]
            except:
                pass

        # 推荐生成
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
            "platoon_ops_diff": features['platoon_ops_diff'],
            "statcast_launch_speed_diff": features['statcast_launch_speed_diff'],
            "statcast_barrel_diff": features['statcast_barrel_diff'],
            "statcast_hard_hit_diff": features['statcast_hard_hit_diff'],
            "statcast_woba_diff": features['statcast_woba_diff'],
            "timezone_diff": features['timezone_diff'],
            "is_day_game": features['is_day_game'],
            "home_back2back": features['home_back2back'],
            "away_back2back": features['away_back2back'],
            "catcher_era_diff": features['catcher_era_diff'],
            "cs_diff": features['cs_diff'],
            "wind_effect": features['wind_effect'],
            "pythag_diff": features['pythag_diff'],
            "log5_prob": features['log5_prob'],
            "lag30_winrate_diff": features['lag30_winrate_diff'],
            "lag30_runs_diff": features['lag30_runs_diff'],
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

    # ========== 正期望值投注评估 ==========
    if filter_value_bets:
        value_bets = filter_value_bets(predictions)
        output['value_bets'] = value_bets

    # ========== 保存历史预测记录 ==========
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
                "elo_diff", "market_prob", "sp_era_diff", "sp_fip_diff",
                "bullpen_ip_diff", "rest_diff", "park_factor",
                "platoon_ops_diff",
                "statcast_launch_speed_diff", "statcast_barrel_diff",
                "statcast_hard_hit_diff", "statcast_woba_diff",
                "timezone_diff", "is_day_game",
                "home_back2back", "away_back2back",
                "catcher_era_diff", "cs_diff", "wind_effect",
                "pythag_diff", "log5_prob",
                "lag30_winrate_diff", "lag30_runs_diff",
                "closing_odds"
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
                "",                     # home_win 留空
                p.get("elo_diff", ""),
                p.get("market_prob", ""),
                p.get("sp_era_diff", ""),
                p.get("sp_fip_diff", ""),
                p.get("bullpen_ip_diff", ""),
                p.get("rest_diff", ""),
                p.get("park_factor", ""),
                p.get("platoon_ops_diff", ""),
                p.get("statcast_launch_speed_diff", ""),
                p.get("statcast_barrel_diff", ""),
                p.get("statcast_hard_hit_diff", ""),
                p.get("statcast_woba_diff", ""),
                p.get("timezone_diff", ""),
                p.get("is_day_game", ""),
                p.get("home_back2back", ""),
                p.get("away_back2back", ""),
                p.get("catcher_era_diff", ""),
                p.get("cs_diff", ""),
                p.get("wind_effect", ""),
                p.get("pythag_diff", ""),
                p.get("log5_prob", ""),
                p.get("lag30_winrate_diff", ""),
                p.get("lag30_runs_diff", ""),
                ""                      # closing_odds 留空
            ])
    print(f"历史预测已追加至 {HISTORY_FILE}")

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
