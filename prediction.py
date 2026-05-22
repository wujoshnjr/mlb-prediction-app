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
try:
    from scripts.player_ratings import calculate_pitcher_ratings
except:
    calculate_pitcher_ratings = None
try:
    from scripts.bullpen_availability import calculate_bullpen_availability
except:
    calculate_bullpen_availability = None
try:
    from scripts.elo_momentum import get_elo_momentum
except:
    get_elo_momentum = None
try:
    from scripts.database import init_database, insert_prediction
except:
    init_database = None
    insert_prediction = None

model = None
try:
    import joblib
    model = joblib.load("data/calibrator.pkl")
    print("已加载训练好的集成模型（双阶段校准）")
except:
    print("未找到训练模型，将使用手工集成")

LAST_GAME_FILE = "data/team_last_game.json"
if os.path.exists(LAST_GAME_FILE):
    with open(LAST_GAME_FILE, 'r') as f:
        last_game_dict = json.load(f)
else:
    last_game_dict = {}

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

FEATURE_DIRECTION = {
    'elo_diff': 1, 'market_prob': 1, 'sp_era_diff': -1, 'sp_fip_diff': -1,
    'bullpen_ip_diff': -1, 'rest_diff': 1, 'park_factor': 1,
    'platoon_ops_diff': 1, 'statcast_launch_speed_diff': 1,
    'statcast_barrel_diff': 1, 'statcast_hard_hit_diff': 1,
    'statcast_woba_diff': 1, 'timezone_diff': -1, 'is_day_game': 0,
    'home_back2back': -1, 'away_back2back': 1, 'catcher_era_diff': -1,
    'cs_diff': 1, 'wind_effect': 1, 'temp_effect': 1,
    'precip_effect': -1, 'injury_diff': -1, 'pythag_diff': 1,
    'log5_prob': 1, 'lag30_winrate_diff': 1, 'lag30_runs_diff': 1,
    'pitch_movement_diff': 1,
    'k_pct_diff': -1, 'bb_pct_diff': 1, 'avg_bat_speed_diff': 1,
    'pitcher_rating_diff': 1, 'dynamic_park_factor': 1, 'odds_change': 1,
    'zone_size': 1, 'k_rate': 1,
    'bullpen_availability_diff': 1,
    'elo_momentum_7d': 1, 'elo_momentum_30d': 1,
    'barrel_pa_diff': 1, 'hardhit_pa_diff': 1,
    'swing_miss_diff': 1, 'csw_diff': 1, 'barrel_bb_pct_diff': -1,
}

def implied_prob(odds):
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

    if elo_system is None:
        if MLBElosystem:
            elo_system = MLBElosystem()
            print("ELO 系统已加载")
        else:
            print("ELO 模块未安装，将使用基础预测")
            elo_system = None

    try:
        from scripts.elo_momentum import save_elo_snapshot
        save_elo_snapshot()
    except: pass

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
    for col in ['runs_scored', 'runs_allowed', 'home_k_pct', 'home_bb_pct', 'away_k_pct', 'away_bb_pct']:
        if col not in teams_df.columns:
            teams_df[col] = 400 if 'runs' in col else (0.2 if 'k' in col else 0.08)

    odds_df = pd.DataFrame(data.get('odds_data', []))
    odds_dict = {}
    odds_source = "bet365,draftkings" if not odds_df.empty else "none"
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

    bullpen_availability_dict = {}
    if calculate_bullpen_availability:
        bullpen_availability_dict = calculate_bullpen_availability(bullpen_df)

    platoon_df = pd.DataFrame(data.get('platoon', []))
    platoon_dict = {}
    if not platoon_df.empty:
        for _, row in platoon_df.iterrows():
            team = row['team_name']; split = row['split']
            if team not in platoon_dict: platoon_dict[team] = {}
            platoon_dict[team][split] = {'ops': float(row.get('ops', 0.700)) if row.get('ops') else 0.700}

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

    pitch_movement_dict = {}
    if not savant_df.empty and 'pfx_x' in savant_df.columns:
        pitch_df = savant_df[['home_team', 'pfx_x', 'pfx_z', 'release_spin_rate']].dropna()
        if not pitch_df.empty:
            team_pitch = pitch_df.groupby('home_team').agg(
                avg_pfx_x=('pfx_x', 'mean'), avg_pfx_z=('pfx_z', 'mean'), avg_spin_rate=('release_spin_rate', 'mean')
            ).reset_index().rename(columns={'home_team': 'team_name'})
            for _, row in team_pitch.iterrows():
                pitch_movement_dict[row['team_name']] = row.to_dict()

    bat_speed_dict = {}
    if not savant_df.empty and 'bat_speed' in savant_df.columns:
        bat_speed_df = savant_df[['home_team', 'bat_speed']].dropna()
        if not bat_speed_df.empty:
            team_bat = bat_speed_df.groupby('home_team')['bat_speed'].mean().reset_index().rename(columns={'home_team': 'team_name'})
            for _, row in team_bat.iterrows():
                bat_speed_dict[row['team_name']] = row['bat_speed']

    pitcher_rating_dict = {}
    if calculate_pitcher_ratings and not savant_df.empty:
        pitcher_rating_dict = calculate_pitcher_ratings(savant_df)

    swing_miss_dict = {}
    csw_dict = {}
    barrel_against_dict = {}
    if not savant_df.empty and 'whiff' in savant_df.columns:
        pitcher_team_col = 'home_team'
        whiff_rate = savant_df.groupby(pitcher_team_col)['whiff'].mean().reset_index()
        whiff_rate.rename(columns={pitcher_team_col: 'team_name', 'whiff': 'whiff_rate'}, inplace=True)
        for _, row in whiff_rate.iterrows():
            swing_miss_dict[row['team_name']] = row['whiff_rate']
        if 'csw' in savant_df.columns:
            csw_rate = savant_df.groupby(pitcher_team_col)['csw'].mean().reset_index()
            csw_rate.rename(columns={pitcher_team_col: 'team_name', 'csw': 'csw_rate'}, inplace=True)
            for _, row in csw_rate.iterrows():
                csw_dict[row['team_name']] = row['csw_rate']
        faced = savant_df.groupby(pitcher_team_col).size().reset_index(name='faced')
        barrel_count = savant_df[savant_df['barrel'] == 1].groupby(pitcher_team_col).size().reset_index(name='barrel_count')
        barrel_rate = faced.merge(barrel_count, on=pitcher_team_col, how='left')
        barrel_rate['barrel_count'] = barrel_rate['barrel_count'].fillna(0)
        barrel_rate['barrel_rate_against'] = barrel_rate['barrel_count'] / barrel_rate['faced']
        barrel_rate.rename(columns={pitcher_team_col: 'team_name'}, inplace=True)
        for _, row in barrel_rate.iterrows():
            barrel_against_dict[row['team_name']] = row['barrel_rate_against']

    weather_df = pd.DataFrame(data.get('openmeteo_weather', []))
    avg_wind_speed = weather_df['wind_speed'].mean() if not weather_df.empty else 0
    avg_wind_dir = weather_df['wind_direction'].mean() if not weather_df.empty else 0
    avg_temp = weather_df['temperature_2m'].mean() if not weather_df.empty else 20.0
    avg_precip = weather_df['precipitation'].mean() if not weather_df.empty else 0.0

    injuries_df = pd.DataFrame(data.get('injuries', []))
    injury_index = {}
    if not injuries_df.empty:
        def injury_severity(status):
            if not isinstance(status, str): return 0.5
            sl = status.lower()
            if '60-day' in sl or '60 day' in sl: return 2.0
            elif '10-day' in sl or '15-day' in sl: return 1.0
            else: return 0.5
        injuries_df['severity_score'] = injuries_df['status'].apply(injury_severity)
        team_injury = injuries_df.groupby('team_name')['severity_score'].sum().reset_index()
        for _, row in team_injury.iterrows():
            injury_index[row['team_name']] = row['severity_score']

    umpire_df = pd.DataFrame(data.get('umpires', []))
    umpire_dict = {}
    if not umpire_df.empty:
        for _, row in umpire_df.iterrows():
            umpire_dict[row['game_id']] = row.to_dict()

    historical_df = None
    hist_dir = "data/historical"
    if os.path.exists(hist_dir):
        hist_files = [os.path.join(hist_dir, f) for f in os.listdir(hist_dir) if f.endswith(".parquet")]
        if hist_files:
            try:
                historical_df = pd.concat([pd.read_parquet(f) for f in hist_files], ignore_index=True)
            except: pass

    team_name_map = {
        "Cleveland Guardians": "Guardians", "Detroit Tigers": "Tigers", "Tampa Bay Rays": "Rays",
        "Baltimore Orioles": "Orioles", "Philadelphia Phillies": "Phillies", "Cincinnati Reds": "Reds",
        "Miami Marlins": "Marlins", "Atlanta Braves": "Braves", "Washington Nationals": "Nationals",
        "New York Mets": "Mets", "New York Yankees": "Yankees", "Toronto Blue Jays": "Blue Jays",
        "Kansas City Royals": "Royals", "Boston Red Sox": "Red Sox", "Minnesota Twins": "Twins",
        "Houston Astros": "Astros", "Chicago Cubs": "Cubs", "Milwaukee Brewers": "Brewers",
        "Colorado Rockies": "Rockies", "Texas Rangers": "Rangers", "Los Angeles Angels": "Angels",
        "Oakland Athletics": "Athletics", "Athletics": "Athletics", "San Diego Padres": "Padres",
        "Los Angeles Dodgers": "Dodgers", "Arizona Diamondbacks": "D-backs", "San Francisco Giants": "Giants",
        "Seattle Mariners": "Mariners", "Chicago White Sox": "White Sox", "Pittsburgh Pirates": "Pirates",
        "St. Louis Cardinals": "Cardinals"
    }

    HIST_FILE = "data/historical_predictions.csv"
    historical_count = 0
    if os.path.exists(HIST_FILE):
        try:
            hist_df = pd.read_csv(HIST_FILE)
            if 'home_win' in hist_df.columns:
                hist_df = hist_df[hist_df['home_win'].notna()]
                historical_count = len(hist_df)
        except: pass

    predictions = []
    for _, game in schedule_df.iterrows():
        home = game.get('home_team', ''); away = game.get('away_team', '')
        home = team_name_map.get(home, home); away = team_name_map.get(away, away)
        if not home or not away or home == 'Unknown' or away == 'Unknown': continue

        home_pct = teams_df[teams_df['name'] == home]['win_pct'].values
        away_pct = teams_df[teams_df['name'] == away]['win_pct'].values
        if len(home_pct) == 0 or len(away_pct) == 0: continue
        home_pct, away_pct = home_pct[0], away_pct[0]

        home_wins = int(teams_df[teams_df['name'] == home]['wins'].values[0])
        home_losses = int(teams_df[teams_df['name'] == home]['losses'].values[0])
        away_wins = int(teams_df[teams_df['name'] == away]['wins'].values[0])
        away_losses = int(teams_df[teams_df['name'] == away]['losses'].values[0])
        home_games = max(home_wins + home_losses, 1)
        away_games = max(away_wins + away_losses, 1)
        home_runs_per_game = float(teams_df[teams_df['name'] == home]['runs_scored'].values[0]) / home_games
        away_runs_per_game = float(teams_df[teams_df['name'] == away]['runs_scored'].values[0]) / away_games

        elo_diff = 0.0
        if elo_system:
            elo_diff = elo_system.elos.get(home, 1500) - elo_system.elos.get(away, 1500) + elo_system.home_adv

        elo_momentum_7d = 0.0; elo_momentum_30d = 0.0
        if get_elo_momentum:
            try:
                home_mom_7 = get_elo_momentum(home, 7)
                away_mom_7 = get_elo_momentum(away, 7)
                elo_momentum_7d = home_mom_7 - away_mom_7
                home_mom_30 = get_elo_momentum(home, 30)
                away_mom_30 = get_elo_momentum(away, 30)
                elo_momentum_30d = home_mom_30 - away_mom_30
            except: pass

        avg_odds = odds_dict.get((home, away), [])
        home_odds = np.mean(avg_odds) if avg_odds else None
        market_prob = implied_prob(home_odds) if home_odds else 0.5

        pitcher_data = pitcher_dict.get(game.get('game_id'))
        sp_era_diff = 0.0; sp_fip_diff = 0.0; home_pitch_hand = "R"; away_pitch_hand = "R"
        if pitcher_data is not None:
            home_era = float(pitcher_data.get('home_era') or 4.5)
            away_era = float(pitcher_data.get('away_era') or 4.5)
            sp_era_diff = home_era - away_era
            home_fip = float(pitcher_data.get('home_fip') or 4.0)
            away_fip = float(pitcher_data.get('away_fip') or 4.0)
            sp_fip_diff = home_fip - away_fip
            home_pitch_hand = pitcher_data.get('home_pitch_hand', 'R')
            away_pitch_hand = pitcher_data.get('away_pitch_hand', 'R')

        rest_diff = 0; timezone_diff = 0; is_day_game = game.get('is_day_game', 0)
        try:
            today = datetime.strptime(date_str, "%Y-%m-%d")
            home_rest = 2; away_rest = 2
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
        except: pass

        bullpen_ip_diff = 0.0; home_back2back = 0; away_back2back = 0
        home_id = TEAM_ID_MAP.get(home); away_id = TEAM_ID_MAP.get(away)
        if home_id and away_id and home_id in bullpen_dict and away_id in bullpen_dict:
            home_ip = bullpen_dict[home_id].get('bullpen_innings')
            away_ip = bullpen_dict[away_id].get('bullpen_innings')
            try:
                home_val = float(home_ip) if home_ip is not None else 0.0
                away_val = float(away_ip) if away_ip is not None else 0.0
                bullpen_ip_diff = home_val - away_val
            except: pass
            home_back2back = int(bullpen_dict[home_id].get('back_to_back', 0))
            away_back2back = int(bullpen_dict[away_id].get('back_to_back', 0))

        bullpen_availability_diff = 0.0
        if bullpen_availability_dict:
            home_avail = bullpen_availability_dict.get(home_id, 50.0)
            away_avail = bullpen_availability_dict.get(away_id, 50.0)
            bullpen_availability_diff = home_avail - away_avail

        from scripts.park_factors import get_park_factor
        static_park = get_park_factor(game.get('venue', ''))
        temp_f = (avg_temp * 9/5) + 32
        temp_effect = (temp_f - 70) / 10 * 0.01
        dynamic_park_factor = static_park * (1 + temp_effect)

        platoon_ops_diff = 0.0
        if not platoon_df.empty:
            home_split = "vsLhp" if home_pitch_hand == "L" else "vsRhp"
            away_split = "vsLhp" if away_pitch_hand == "L" else "vsRhp"
            home_platoon = platoon_dict.get(home, {}).get(home_split, {})
            away_platoon = platoon_dict.get(away, {}).get(away_split, {})
            if home_platoon and away_platoon:
                platoon_ops_diff = home_platoon['ops'] - away_platoon['ops']

        catcher_era_diff = 0.0; cs_diff = 0.0
        if calculate_catcher_effect:
            home_catcher_id = game.get('home_catcher_id')
            away_catcher_id = game.get('away_catcher_id')
            if home_catcher_id and away_catcher_id:
                catcher_era_diff, cs_diff = calculate_catcher_effect(home_catcher_id, away_catcher_id, 2026)

        statcast_launch_speed_diff = 0.0; statcast_barrel_diff = 0.0; statcast_hard_hit_diff = 0.0; statcast_woba_diff = 0.0
        if statcast_team_stats:
            home_hit = statcast_team_stats.get('home'); away_hit = statcast_team_stats.get('away')
            if home_hit is not None and away_hit is not None:
                home_row = home_hit[home_hit['team_name'] == home]
                away_row = away_hit[away_hit['team_name'] == away]
                if not home_row.empty and not away_row.empty:
                    statcast_launch_speed_diff = home_row.iloc[0]['avg_launch_speed'] - away_row.iloc[0]['avg_launch_speed']
                    statcast_barrel_diff = home_row.iloc[0]['barrel_rate'] - away_row.iloc[0]['barrel_rate']
                    statcast_hard_hit_diff = home_row.iloc[0]['hard_hit_rate'] - away_row.iloc[0]['hard_hit_rate']
                    statcast_woba_diff = home_row.iloc[0]['avg_expected_woba'] - away_row.iloc[0]['avg_expected_woba']

        pitch_movement_diff = 0.0
        if pitch_movement_dict:
            home_pitch = pitch_movement_dict.get(home); away_pitch = pitch_movement_dict.get(away)
            if home_pitch and away_pitch:
                home_movement = np.sqrt(home_pitch['avg_pfx_x']**2 + home_pitch['avg_pfx_z']**2)
                away_movement = np.sqrt(away_pitch['avg_pfx_x']**2 + away_pitch['avg_pfx_z']**2)
                pitch_movement_diff = home_movement - away_movement

        home_k_pct = teams_df[teams_df['name'] == home]['home_k_pct'].values[0]
        away_k_pct = teams_df[teams_df['name'] == away]['away_k_pct'].values[0]
        k_pct_diff = home_k_pct - away_k_pct
        home_bb_pct = teams_df[teams_df['name'] == home]['home_bb_pct'].values[0]
        away_bb_pct = teams_df[teams_df['name'] == away]['away_bb_pct'].values[0]
        bb_pct_diff = home_bb_pct - away_bb_pct

        avg_bat_speed_diff = 0.0
        if bat_speed_dict:
            home_bs = bat_speed_dict.get(home); away_bs = bat_speed_dict.get(away)
            if home_bs is not None and away_bs is not None:
                avg_bat_speed_diff = home_bs - away_bs

        pitcher_rating_diff = 0.0
        if pitcher_rating_dict:
            home_rating = pitcher_rating_dict.get(home); away_rating = pitcher_rating_dict.get(away)
            if home_rating is not None and away_rating is not None:
                pitcher_rating_diff = home_rating - away_rating

        swing_miss_diff = 0.0
        if swing_miss_dict:
            h_whiff = swing_miss_dict.get(home); a_whiff = swing_miss_dict.get(away)
            if h_whiff is not None and a_whiff is not None: swing_miss_diff = h_whiff - a_whiff
        csw_diff = 0.0
        if csw_dict:
            h_csw = csw_dict.get(home); a_csw = csw_dict.get(away)
            if h_csw is not None and a_csw is not None: csw_diff = h_csw - a_csw
        barrel_bb_pct_diff = 0.0
        if barrel_against_dict:
            h_barrel = barrel_against_dict.get(home); a_barrel = barrel_against_dict.get(away)
            if h_barrel is not None and a_barrel is not None: barrel_bb_pct_diff = h_barrel - a_barrel

        temp_effect_raw = 0.0
        if avg_temp > 25: temp_effect_raw = 0.02 * (avg_temp - 25)
        precip_effect = -0.01 * avg_precip if avg_precip > 0 else 0.0
        wind_effect = 0.0
        if avg_wind_speed > 10:
            wind_effect = 0.02 * avg_wind_speed * np.sin(np.radians(avg_wind_dir))

        home_injury_impact = injury_index.get(home, 0.0); away_injury_impact = injury_index.get(away, 0.0)
        injury_diff = home_injury_impact - away_injury_impact

        home_runs_scored = float(teams_df[teams_df['name'] == home]['runs_scored'].values[0])
        home_runs_allowed = float(teams_df[teams_df['name'] == home]['runs_allowed'].values[0])
        away_runs_scored = float(teams_df[teams_df['name'] == away]['runs_scored'].values[0])
        away_runs_allowed = float(teams_df[teams_df['name'] == away]['runs_allowed'].values[0])
        home_pythag = home_runs_scored**1.83 / (home_runs_scored**1.83 + home_runs_allowed**1.83) if (home_runs_scored + home_runs_allowed) > 0 else 0.5
        away_pythag = away_runs_scored**1.83 / (away_runs_scored**1.83 + away_runs_allowed**1.83) if (away_runs_scored + away_runs_allowed) > 0 else 0.5
        pythag_diff = home_pythag - away_pythag

        log5_home = (home_pct - home_pct * away_pct) / (home_pct + away_pct - 2 * home_pct * away_pct) if (home_pct + away_pct) > 0 else 0.5

        lag30_winrate_diff = 0.0; lag30_runs_diff = 0.0
        if calculate_lag_features and historical_df is not None:
            try:
                home_winrate_30, away_winrate_30, home_runs_30, away_runs_30 = calculate_lag_features(home, away, historical_df, date_str, days=30)
                lag30_winrate_diff = home_winrate_30 - away_winrate_30
                lag30_runs_diff = home_runs_30 - away_runs_30
            except: pass

        odds_change = 0.0
        if os.path.exists(HIST_FILE):
            try:
                hist = pd.read_csv(HIST_FILE)
                last_odds = hist[(hist['home_team'] == home) & (hist['away_team'] == away)]['home_odds'].dropna().tail(1).values
                if len(last_odds) > 0 and home_odds is not None: odds_change = home_odds - last_odds[0]
            except: pass

        umpire_data = umpire_dict.get(game.get('game_id'), {})
        zone_size = umpire_data.get("zone_size", 1.0)
        k_rate = umpire_data.get("k_rate", 0.0)

        barrel_pa_diff = statcast_barrel_diff
        hardhit_pa_diff = statcast_hard_hit_diff

        features = {
            'elo_diff': round(elo_diff, 3),
            'market_prob': round(market_prob, 3) if market_prob else 0.5,
            'sp_era_diff': round(sp_era_diff, 3),
            'sp_fip_diff': round(sp_fip_diff, 3),
            'bullpen_ip_diff': round(bullpen_ip_diff, 3),
            'rest_diff': rest_diff,
            'park_factor': static_park,
            'dynamic_park_factor': round(dynamic_park_factor, 3),
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
            'temp_effect': round(temp_effect_raw, 4),
            'precip_effect': round(precip_effect, 4),
            'injury_diff': round(injury_diff, 3),
            'pythag_diff': round(pythag_diff, 3),
            'log5_prob': round(log5_home, 3),
            'lag30_winrate_diff': round(lag30_winrate_diff, 3),
            'lag30_runs_diff': round(lag30_runs_diff, 3),
            'pitch_movement_diff': round(pitch_movement_diff, 3),
            'k_pct_diff': round(k_pct_diff, 3),
            'bb_pct_diff': round(bb_pct_diff, 3),
            'avg_bat_speed_diff': round(avg_bat_speed_diff, 3),
            'pitcher_rating_diff': round(pitcher_rating_diff, 3),
            'odds_change': round(odds_change, 4),
            'zone_size': round(zone_size, 3),
            'k_rate': round(k_rate, 3),
            'bullpen_availability_diff': round(bullpen_availability_diff, 3),
            'elo_momentum_7d': round(elo_momentum_7d, 3),
            'elo_momentum_30d': round(elo_momentum_30d, 3),
            'barrel_pa_diff': round(barrel_pa_diff, 3),
            'hardhit_pa_diff': round(hardhit_pa_diff, 3),
            'swing_miss_diff': round(swing_miss_diff, 3),
            'csw_diff': round(csw_diff, 3),
            'barrel_bb_pct_diff': round(barrel_bb_pct_diff, 3),
            'home_winrate': round(home_pct, 3),
            'away_winrate': round(away_pct, 3)
        }

        weights = {'pct': 0.25, 'elo': 0.35, 'market': 0.40}
        if elo_system is None:
            weights['elo'] = 0; weights['pct'] += 0.175; weights['market'] += 0.175
        if market_prob is None or home_odds is None:
            weights['market'] = 0
            if elo_system is not None: weights['elo'] += 0.20; weights['pct'] += 0.20
            else: weights['pct'] += 0.40

        elo_prob = 1 / (1 + 10 ** (-elo_diff / 400)) if elo_system else 0.5
        manual_pred = home_pct * weights['pct'] + elo_prob * weights['elo'] + (market_prob or 0.5) * weights['market']
        sp_adj = -0.07 * sp_era_diff
        manual_pred = min(0.95, max(0.05, manual_pred + sp_adj))

        # 无赔率手工预测
        if elo_system:
            no_odds_pred = home_pct * 0.4 + elo_prob * 0.6
        else:
            no_odds_pred = home_pct
        no_odds_pred = min(0.95, max(0.05, no_odds_pred + sp_adj))

        ml_pred = None
        xgb_pred = None
        lgb_pred = None
        if model is not None:
            feature_array = np.array([[
                features['elo_diff'], features['market_prob'], features['sp_era_diff'],
                features['sp_fip_diff'], features['bullpen_ip_diff'], features['rest_diff'],
                features['dynamic_park_factor'], features['platoon_ops_diff'],
                features['statcast_launch_speed_diff'], features['statcast_barrel_diff'],
                features['statcast_hard_hit_diff'], features['statcast_woba_diff'],
                features['timezone_diff'], features['is_day_game'],
                features['home_back2back'], features['away_back2back'],
                features['catcher_era_diff'], features['cs_diff'],
                features['wind_effect'], features['temp_effect'],
                features['precip_effect'], features['injury_diff'],
                features['pythag_diff'], features['log5_prob'],
                features['lag30_winrate_diff'], features['lag30_runs_diff'],
                features['pitch_movement_diff'],
                features['k_pct_diff'], features['bb_pct_diff'],
                features['avg_bat_speed_diff'],
                features['pitcher_rating_diff'],
                features['odds_change'],
                features['zone_size'], features['k_rate'],
                features['bullpen_availability_diff'],
                features['elo_momentum_7d'], features['elo_momentum_30d'],
                features['barrel_pa_diff'], features['hardhit_pa_diff'],
                features['swing_miss_diff'], features['csw_diff'],
                features['barrel_bb_pct_diff']
            ]])
            try:
                ml_pred = model.predict_proba(feature_array)[0, 1]
                ml_pred = min(0.95, max(0.05, ml_pred))
                if hasattr(model, 'platt') and hasattr(model.platt, 'estimators_'):
                    xgb_pred = model.platt.estimators_[0].predict_proba(feature_array)[0, 1]
                    lgb_pred = model.platt.estimators_[1].predict_proba(feature_array)[0, 1]
            except:
                ml_pred = None

        if ml_pred is not None and historical_count > 100:
            ml_weight = min(0.5, historical_count / 1000)
            pred_home = (1 - ml_weight) * manual_pred + ml_weight * ml_pred
        else:
            pred_home = manual_pred

        pred_away = 1 - pred_home

        pred_uncertainty = 0.0
        if xgb_pred is not None and lgb_pred is not None:
            pred_uncertainty = abs(xgb_pred - lgb_pred)

        pred_std = 0.10
        if historical_count > 10:
            try:
                hist_df = pd.read_csv(HIST_FILE)
                hist_df = hist_df.dropna(subset=['home_win', 'pred_home_win'])
                if len(hist_df) > 10:
                    similar = hist_df[(hist_df['pred_home_win'] >= pred_home - 0.05) & (hist_df['pred_home_win'] <= pred_home + 0.05)]
                    if len(similar) > 3:
                        pred_std = max(0.02, similar['home_win'].std())
            except: pass
        z = 1.28
        ci_lower = max(0.01, pred_home - z * pred_std)
        ci_upper = min(0.99, pred_home + z * pred_std)

        kelly_ml = 0; kelly_ml_away = 0
        if home_odds:
            kelly_ml = kelly_criterion(pred_home, home_odds)
            kelly_ml_away = kelly_criterion(pred_away, 1 / (1 - implied_prob(home_odds)) if implied_prob(home_odds) and implied_prob(home_odds) < 1 else None)

        sim = None; home_cover = away_cover = over_prob = under_prob = None; total_mean = diff_mean = None; ci_diff = []
        if MonteCarloSimulator:
            try:
                hr = home_runs_per_game; ar = away_runs_per_game
                env_adj = 1 + wind_effect + temp_effect_raw + precip_effect
                hr_adj = hr * dynamic_park_factor * env_adj
                ar_adj = ar * dynamic_park_factor * env_adj
                sim = MonteCarloSimulator(hr_adj, ar_adj, n_simulations=5000)
                sim.simulate()
                home_cover, away_cover = sim.spread_prob(-1.5)
                over_prob, under_prob, _ = sim.total_prob(8.5)
                total_mean = round(sim.results['total_runs'].mean(), 1)
                diff_mean = round(sim.results['run_diff'].mean(), 1)
                ci_diff = [round(x, 1) for x in sim.confidence_interval()]
            except: pass

        ml_rec = "PASS"
        if kelly_ml > 0.05: ml_rec = f"Bet {home} ({pred_home:.1%}, {kelly_ml:.1%} Kelly)"
        elif kelly_ml_away > 0.05: ml_rec = f"Bet {away} ({pred_away:.1%}, {kelly_ml_away:.1%} Kelly)"

        spread_rec = "PASS"
        if home_cover is not None and home_cover > 0.55: spread_rec = f"Bet {home} -1.5 ({home_cover:.1%} cover)"
        elif away_cover is not None and away_cover > 0.55: spread_rec = f"Bet {away} +1.5 ({away_cover:.1%} cover)"

        total_rec = "PASS"
        if over_prob is not None and over_prob > 0.55: total_rec = f"Bet OVER 8.5 ({over_prob:.1%})"
        elif under_prob is not None and under_prob > 0.55: total_rec = f"Bet UNDER 8.5 ({under_prob:.1%})"

        nrfi_prob = 0.5
        if pitcher_data is not None:
            home_first_era = float(pitcher_data.get('home_first_era', 4.5) or 4.5)
            away_first_era = float(pitcher_data.get('away_first_era', 4.5) or 4.5)
            nrfi_prob = max(0.3, min(0.7, 0.5 + (4.5 - (home_first_era + away_first_era) / 2) * 0.08))
        nrfi_rec = f"NRFI ({nrfi_prob:.1%})" if nrfi_prob > 0.55 else f"YRFI ({(1-nrfi_prob):.1%})"

        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = []
        for name, val in sorted_features[:5]:
            direction = FEATURE_DIRECTION.get(name, 0)
            if direction == 1: arrow = "▲" if val > 0 else "▼"
            elif direction == -1: arrow = "▼" if val > 0 else "▲"
            else: arrow = "•"
            top_features.append(f"{name}={val}{arrow}")

        predictions.append({
            "game_id": game.get("game_id"),
            "game_date": game.get("game_date"),
            "home_team": home, "away_team": away,
            "status": game.get("status"), "venue": game.get("venue", ""),
            "predicted_home_win_pct": round(pred_home, 3),
            "predicted_away_win_pct": round(pred_away, 3),
            "home_odds": home_odds,
            "manual_no_odds_pred": round(no_odds_pred, 3),
            "elo_home": elo_system.elos.get(home, 1500) if elo_system else 1500,
            "elo_away": elo_system.elos.get(away, 1500) if elo_system else 1500,
            "moneyline_recommendation": ml_rec,
            "spread_recommendation": spread_rec,
            "total_recommendation": total_rec,
            "nrfi_recommendation": nrfi_rec,
            "nrfi_prob": round(nrfi_prob, 3),
            "pred_uncertainty": round(pred_uncertainty, 4),
            "ci_lower": round(ci_lower, 3),
            "ci_upper": round(ci_upper, 3),
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
            "dynamic_park_factor": features['dynamic_park_factor'],
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
            "temp_effect": features['temp_effect'],
            "precip_effect": features['precip_effect'],
            "injury_diff": features['injury_diff'],
            "pythag_diff": features['pythag_diff'],
            "log5_prob": features['log5_prob'],
            "lag30_winrate_diff": features['lag30_winrate_diff'],
            "lag30_runs_diff": features['lag30_runs_diff'],
            "pitch_movement_diff": features['pitch_movement_diff'],
            "k_pct_diff": features['k_pct_diff'],
            "bb_pct_diff": features['bb_pct_diff'],
            "avg_bat_speed_diff": features['avg_bat_speed_diff'],
            "pitcher_rating_diff": features['pitcher_rating_diff'],
            "odds_change": features['odds_change'],
            "zone_size": features['zone_size'],
            "k_rate": features['k_rate'],
            "bullpen_availability_diff": features['bullpen_availability_diff'],
            "elo_momentum_7d": features['elo_momentum_7d'],
            "elo_momentum_30d": features['elo_momentum_30d'],
            "barrel_pa_diff": features['barrel_pa_diff'],
            "hardhit_pa_diff": features['hardhit_pa_diff'],
            "swing_miss_diff": features['swing_miss_diff'],
            "csw_diff": features['csw_diff'],
            "barrel_bb_pct_diff": features['barrel_bb_pct_diff'],
            "home_winrate": features['home_winrate'],
            "away_winrate": features['away_winrate'],
            "top_features": top_features,
            "market_divergence": 1 if abs(pred_home - market_prob) > 0.15 else 0,
            "odds_source": odds_source
        })

    power_rankings = teams_df.sort_values('win_pct', ascending=False).to_dict('records')

    output = {
        "generated_at": datetime.now().isoformat(),
        "power_rankings": power_rankings,
        "elo_ratings": {k: round(v, 1) for k, v in elo_system.elos.items()} if elo_system else {},
        "today_predictions": predictions,
        "bet_summary": {
            "moneyline_bets": [p for p in predictions if p['moneyline_recommendation'] != 'PASS'],
            "spread_bets": [p for p in predictions if p['spread_recommendation'] != 'PASS'],
            "total_bets": [p for p in predictions if p['total_recommendation'] != 'PASS'],
            "nrfi_bets": [p for p in predictions if 'NRFI' in p.get('nrfi_recommendation', '')]
        },
        "errors": errors
    }

    if filter_value_bets:
        value_bets = filter_value_bets(predictions)
        output['value_bets'] = value_bets

    # 写入数据库
    db_available = init_database is not None and insert_prediction is not None
    if db_available:
        try:
            init_database()
        except:
            db_available = False
    if db_available:
        for p in predictions:
            db_data = {
                "game_id": p.get("game_id"), "game_date": p.get("game_date"),
                "home_team": p.get("home_team"), "away_team": p.get("away_team"),
                "pred_home_win": p.get("predicted_home_win_pct"), "home_odds": p.get("home_odds"),
                "elo_home": p.get("elo_home"), "elo_away": p.get("elo_away"),
                "ml_rec": p.get("moneyline_recommendation"), "spread_rec": p.get("spread_recommendation"),
                "total_rec": p.get("total_recommendation"), "nrfi_rec": p.get("nrfi_recommendation"),
                "nrfi_prob": p.get("nrfi_prob"), "pred_uncertainty": p.get("pred_uncertainty"),
                "kelly_fraction": p.get("kelly_fraction"),
                "manual_no_odds_pred": p.get("manual_no_odds_pred"),
                "elo_diff": p.get("elo_diff"), "market_prob": p.get("market_prob"),
                "sp_era_diff": p.get("sp_era_diff"), "sp_fip_diff": p.get("sp_fip_diff"),
                "bullpen_ip_diff": p.get("bullpen_ip_diff"), "rest_diff": p.get("rest_diff"),
                "park_factor": p.get("park_factor"), "dynamic_park_factor": p.get("dynamic_park_factor"),
                "platoon_ops_diff": p.get("platoon_ops_diff"),
                "statcast_launch_speed_diff": p.get("statcast_launch_speed_diff"),
                "statcast_barrel_diff": p.get("statcast_barrel_diff"),
                "statcast_hard_hit_diff": p.get("statcast_hard_hit_diff"),
                "statcast_woba_diff": p.get("statcast_woba_diff"),
                "timezone_diff": p.get("timezone_diff"), "is_day_game": p.get("is_day_game"),
                "home_back2back": p.get("home_back2back"), "away_back2back": p.get("away_back2back"),
                "catcher_era_diff": p.get("catcher_era_diff"), "cs_diff": p.get("cs_diff"),
                "wind_effect": p.get("wind_effect"), "temp_effect": p.get("temp_effect"),
                "precip_effect": p.get("precip_effect"), "injury_diff": p.get("injury_diff"),
                "pythag_diff": p.get("pythag_diff"), "log5_prob": p.get("log5_prob"),
                "lag30_winrate_diff": p.get("lag30_winrate_diff"), "lag30_runs_diff": p.get("lag30_runs_diff"),
                "pitch_movement_diff": p.get("pitch_movement_diff"),
                "k_pct_diff": p.get("k_pct_diff"), "bb_pct_diff": p.get("bb_pct_diff"),
                "avg_bat_speed_diff": p.get("avg_bat_speed_diff"),
                "pitcher_rating_diff": p.get("pitcher_rating_diff"),
                "odds_change": p.get("odds_change"),
                "zone_size": p.get("zone_size"), "k_rate": p.get("k_rate"),
                "bullpen_availability_diff": p.get("bullpen_availability_diff"),
                "elo_momentum_7d": p.get("elo_momentum_7d"), "elo_momentum_30d": p.get("elo_momentum_30d"),
                "barrel_pa_diff": p.get("barrel_pa_diff"), "hardhit_pa_diff": p.get("hardhit_pa_diff"),
                "swing_miss_diff": p.get("swing_miss_diff"), "csw_diff": p.get("csw_diff"),
                "barrel_bb_pct_diff": p.get("barrel_bb_pct_diff"),
                "closing_odds": None,
                "top_features": ";".join(p.get("top_features", [])) if isinstance(p.get("top_features"), list) else p.get("top_features", ""),
                "market_divergence": p.get("market_divergence", 0),
                "odds_source": p.get("odds_source", "")
            }
            try:
                insert_prediction(db_data)
            except:
                db_available = False
                break

    # CSV 双写
    HISTORY_FILE = "data/historical_predictions.csv"
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "game_id", "game_date", "home_team", "away_team",
                "pred_home_win", "home_odds", "elo_home", "elo_away",
                "ml_rec", "spread_rec", "total_rec", "nrfi_rec", "nrfi_prob", "pred_uncertainty",
                "manual_no_odds_pred", "ci_lower", "ci_upper",
                "kelly_fraction", "home_win",
                "elo_diff", "market_prob", "sp_era_diff", "sp_fip_diff",
                "bullpen_ip_diff", "rest_diff", "park_factor",
                "dynamic_park_factor",
                "platoon_ops_diff",
                "statcast_launch_speed_diff", "statcast_barrel_diff",
                "statcast_hard_hit_diff", "statcast_woba_diff",
                "timezone_diff", "is_day_game",
                "home_back2back", "away_back2back",
                "catcher_era_diff", "cs_diff", "wind_effect",
                "temp_effect", "precip_effect", "injury_diff",
                "pythag_diff", "log5_prob",
                "lag30_winrate_diff", "lag30_runs_diff",
                "pitch_movement_diff",
                "k_pct_diff", "bb_pct_diff", "avg_bat_speed_diff",
                "pitcher_rating_diff", "odds_change",
                "zone_size", "k_rate", "bullpen_availability_diff",
                "elo_momentum_7d", "elo_momentum_30d", "barrel_pa_diff", "hardhit_pa_diff",
                "swing_miss_diff", "csw_diff", "barrel_bb_pct_diff",
                "closing_odds", "top_features", "market_divergence", "odds_source"
            ])
        for p in predictions:
            writer.writerow([
                p.get("game_id", ""), p.get("game_date", ""), p.get("home_team", ""), p.get("away_team", ""),
                p.get("predicted_home_win_pct", ""), p.get("home_odds", ""), p.get("elo_home", ""), p.get("elo_away", ""),
                p.get("moneyline_recommendation", ""), p.get("spread_recommendation", ""), p.get("total_recommendation", ""),
                p.get("nrfi_recommendation", ""), p.get("nrfi_prob", ""), p.get("pred_uncertainty", ""),
                p.get("manual_no_odds_pred", ""), p.get("ci_lower", ""), p.get("ci_upper", ""),
                p.get("kelly_fraction", ""), "",
                p.get("elo_diff", ""), p.get("market_prob", ""), p.get("sp_era_diff", ""), p.get("sp_fip_diff", ""),
                p.get("bullpen_ip_diff", ""), p.get("rest_diff", ""), p.get("park_factor", ""),
                p.get("dynamic_park_factor", ""),
                p.get("platoon_ops_diff", ""), p.get("statcast_launch_speed_diff", ""), p.get("statcast_barrel_diff", ""),
                p.get("statcast_hard_hit_diff", ""), p.get("statcast_woba_diff", ""),
                p.get("timezone_diff", ""), p.get("is_day_game", ""), p.get("home_back2back", ""), p.get("away_back2back", ""),
                p.get("catcher_era_diff", ""), p.get("cs_diff", ""), p.get("wind_effect", ""),
                p.get("temp_effect", ""), p.get("precip_effect", ""), p.get("injury_diff", ""),
                p.get("pythag_diff", ""), p.get("log5_prob", ""),
                p.get("lag30_winrate_diff", ""), p.get("lag30_runs_diff", ""),
                p.get("pitch_movement_diff", ""),
                p.get("k_pct_diff", ""), p.get("bb_pct_diff", ""), p.get("avg_bat_speed_diff", ""),
                p.get("pitcher_rating_diff", ""), p.get("odds_change", ""),
                p.get("zone_size", ""), p.get("k_rate", ""), p.get("bullpen_availability_diff", ""),
                p.get("elo_momentum_7d", ""), p.get("elo_momentum_30d", ""), p.get("barrel_pa_diff", ""), p.get("hardhit_pa_diff", ""),
                p.get("swing_miss_diff", ""), p.get("csw_diff", ""), p.get("barrel_bb_pct_diff", ""),
                "",
                ";".join(p.get("top_features", [])) if isinstance(p.get("top_features"), list) else p.get("top_features", ""),
                p.get("market_divergence", 0),
                p.get("odds_source", "")
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
