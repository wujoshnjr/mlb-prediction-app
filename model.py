# model.py
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import math

# 防御性导入：任何一个模块失败都不会影响主服务启动
fetch_mlb_statsapi = None
fetch_savant_statcast = None
fetch_retrosheet = None
fetch_pybaseball = None
fetch_sportsipy = None
fetch_openmeteo = None
fetch_balldontlie = None
fetch_odds = None
fetch_probable_pitchers = None
fetch_injuries = None
fetch_bullpen_stats = None
fetch_platoon_splits = None
fetch_umpire_data = None

# ---------- 特征相关模块（防御性）----------
EloSystem = None
get_elo_momentum = None
get_lag_features = None
get_catcher_effect = None
get_park_factor = None
get_pitch_type_matchup_score = None
bradley_terry_strength = None
get_bullpen_availability = None
get_pitcher_rating = None

try:
    from scripts.elo import EloSystem
except Exception:
    pass

try:
    from scripts.elo_momentum import get_elo_momentum
except Exception:
    pass

try:
    from scripts.lag_features import get_lag_features
except Exception:
    pass

try:
    from scripts.catcher_utils import get_catcher_effect
except Exception:
    pass

try:
    from scripts.park_factors import get_park_factor
except Exception:
    pass

try:
    from scripts.pitch_type_matchup import get_pitch_type_matchup_score
except Exception:
    pass

try:
    from scripts.bradley_terry import bradley_terry_strength
except Exception:
    pass

try:
    from scripts.bullpen_availability import get_bullpen_availability
except Exception:
    pass

try:
    from scripts.player_ratings import get_pitcher_rating
except Exception:
    pass

# Glicko2 相关
Glicko2League = None
load_glicko2_league = None
try:
    from scripts.glicko2_ratings import Glicko2League
    from scripts.rating_updater import load_glicko2_league
except Exception:
    pass

# Pitch Matchup
add_matchup_features = None
get_matchup_lookup = None
try:
    from scripts.batter_vs_pitch_client import add_matchup_features, get_matchup_lookup
except Exception:
    pass

# 盘口曲线
extract_odds_curve_features = None
try:
    from scripts.odds_client import extract_odds_curve_features
except Exception:
    pass

# 配置开关
try:
    import config
except ImportError:
    # 如果没有 config.py，提供默认值
    class config:
        RATINGS_ENGINE = 'elo'
        FEATURE_USE_PITCH_MATCHUP = False
        ODDS_USE_CURVE_FEATURES = False

# ---------- 原始客户端导入 ----------
try:
    from scripts.mlb_stats_client import fetch_mlb_statsapi
except Exception as e:
    print(f"Warning: Failed to import mlb_stats_client: {e}")

try:
    from scripts.savant_client import fetch_savant_statcast
except Exception as e:
    print(f"Warning: Failed to import savant_client: {e}")

try:
    from scripts.retro_client import fetch_retrosheet
except Exception as e:
    print(f"Warning: Failed to import retro_client: {e}")

try:
    from scripts.pybaseball_client import fetch_pybaseball
except Exception as e:
    print(f"Warning: Failed to import pybaseball_client: {e}")

try:
    from scripts.sportsipy_client import fetch_sportsipy
except Exception as e:
    print(f"Warning: Failed to import sportsipy_client: {e}")

try:
    from scripts.openmeteo_client import fetch_openmeteo
except Exception as e:
    print(f"Warning: Failed to import openmeteo_client: {e}")

try:
    from scripts.balldontlie_client import fetch_balldontlie
except Exception as e:
    print(f"Warning: Failed to import balldontlie_client: {e}")

try:
    from scripts.odds_client import fetch_odds
except Exception as e:
    print(f"Warning: Failed to import odds_client: {e}")

try:
    from scripts.pitcher_client import fetch_probable_pitchers
except Exception as e:
    print(f"Warning: Failed to import pitcher_client: {e}")

try:
    from scripts.injury_client import fetch_injuries
except Exception as e:
    print(f"Warning: Failed to import injury_client: {e}")

try:
    from scripts.bullpen_client import fetch_bullpen_stats
except Exception as e:
    print(f"Warning: Failed to import bullpen_client: {e}")

try:
    from scripts.platoon_client import fetch_platoon_splits
except Exception as e:
    print(f"Warning: Failed to import platoon_client: {e}")

try:
    from scripts.umpire_client import fetch_umpire_data
except Exception as e:
    print(f"Warning: Failed to import umpire_client: {e}")


# 如果 EloSystem 导入失败，使用一个简易替代
if EloSystem is None:
    class DummyElo:
        def get_rating(self, team):
            return 1500
    EloSystem = DummyElo


class UnifiedSportsModel:
    def __init__(self):
        raw_ball = os.getenv("BALLDONTLIE_API_KEY", "") or ""
        raw_odds = os.getenv("ODDS_API_KEY", "") or ""
        self.ball_api_key = raw_ball.strip().replace("\n", "").replace("\r", "")
        self.odds_api_key = raw_odds.strip().replace("\n", "").replace("\r", "")
        self.elo_system = EloSystem()
        self._glicko_league = None
        self._matchup_lookup = None

    def gather_all_data(self, date_str: str = None) -> dict:
        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')

        errors = []
        result = {
            'date': date_str,
            'mlb_statsapi': [], 'savant_statcast': [], 'retrosheet': [],
            'pybaseball_statcast': [], 'pybaseball_batting': [], 'pybaseball_pitching': [],
            'sportsipy_teams': [], 'sportsipy_player': {},
            'openmeteo_weather': [], 'balldontlie_teams': [], 'odds_data': [],
            'pitchers': [], 'injuries': [], 'bullpen': [], 'platoon': [],
            'umpires': [],
            'errors': errors
        }

        def safe_call(func, name, *args):
            if func is None:
                errors.append(f"{name} module not loaded.")
                return pd.DataFrame() if name not in ("pybaseball", "sportsipy") else {}
            try:
                res = func(*args)
                return res
            except Exception as e:
                errors.append(f"{name} fetch error: {e}")
                return pd.DataFrame() if name not in ("pybaseball", "sportsipy") else {}

        mlb_stats = safe_call(fetch_mlb_statsapi, "mlb_statsapi", date_str, errors)
        savant = safe_call(fetch_savant_statcast, "savant_statcast", date_str, errors)
        retro = safe_call(fetch_retrosheet, "retrosheet", date_str, errors)
        pyb = safe_call(fetch_pybaseball, "pybaseball", date_str, errors)
        sportsipy = safe_call(fetch_sportsipy, "sportsipy", date_str, errors)
        openmeteo = safe_call(fetch_openmeteo, "openmeteo", date_str, errors)
        balldontlie = safe_call(fetch_balldontlie, "balldontlie", self.ball_api_key, date_str, errors)
        odds = safe_call(fetch_odds, "odds", self.odds_api_key, date_str, errors)
        pitchers = safe_call(fetch_probable_pitchers, "pitchers", date_str, errors)
        injuries = safe_call(fetch_injuries, "injuries", date_str, errors)
        bullpen = safe_call(fetch_bullpen_stats, "bullpen", date_str, errors)
        platoon = safe_call(fetch_platoon_splits, "platoon", 2026, errors)
        umpire = safe_call(fetch_umpire_data, "umpires", date_str, errors)

        result['mlb_statsapi'] = mlb_stats.to_dict(orient='records') if not mlb_stats.empty else []
        result['savant_statcast'] = savant.to_dict(orient='records') if not savant.empty else []
        result['retrosheet'] = retro.to_dict(orient='records') if not retro.empty else []
        if isinstance(pyb, dict):
            result['pybaseball_statcast'] = pyb.get('statcast_recent', pd.DataFrame()).to_dict(orient='records') if not pyb.get('statcast_recent', pd.DataFrame()).empty else []
            result['pybaseball_batting'] = pyb.get('batting_leaders', pd.DataFrame()).to_dict(orient='records') if not pyb.get('batting_leaders', pd.DataFrame()).empty else []
            result['pybaseball_pitching'] = pyb.get('pitching_leaders', pd.DataFrame()).to_dict(orient='records') if not pyb.get('pitching_leaders', pd.DataFrame()).empty else []
        if isinstance(sportsipy, dict):
            result['sportsipy_teams'] = sportsipy.get('teams', pd.DataFrame()).to_dict(orient='records') if not sportsipy.get('teams', pd.DataFrame()).empty else []
            result['sportsipy_player'] = sportsipy.get('player_example', {})
        result['openmeteo_weather'] = openmeteo.to_dict(orient='records') if not openmeteo.empty else []
        result['balldontlie_teams'] = balldontlie.to_dict(orient='records') if not balldontlie.empty else []
        result['odds_data'] = odds.to_dict(orient='records') if not odds.empty else []
        result['pitchers'] = pitchers.to_dict(orient='records') if not pitchers.empty else []
        result['injuries'] = injuries.to_dict(orient='records') if not injuries.empty else []
        result['bullpen'] = bullpen.to_dict(orient='records') if not bullpen.empty else []
        result['platoon'] = platoon.to_dict(orient='records') if not platoon.empty else []
        result['umpires'] = umpire.to_dict(orient='records') if not umpire.empty else []

        if os.path.isfile('report'):
            os.remove('report')
        os.makedirs('report', exist_ok=True)
        with open(f'report/{date_str}.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)

        return result

    # ---------- 特征构建 ----------
    def get_glicko_league(self):
        if config.RATINGS_ENGINE == 'glicko2' and load_glicko2_league is not None:
            if self._glicko_league is None:
                self._glicko_league = load_glicko2_league()
            return self._glicko_league
        return None

    def get_matchup_lookup(self):
        if config.FEATURE_USE_PITCH_MATCHUP and get_matchup_lookup is not None:
            if self._matchup_lookup is None:
                self._matchup_lookup = get_matchup_lookup()
            return self._matchup_lookup
        return None

    def build_game_features(self, game: dict) -> dict:
        home_team = game['home_team']
        away_team = game['away_team']
        game_date = game['date']
        venue = game.get('venue', '')
        game_id = game.get('game_id', f"{home_team}_{away_team}_{game_date}")

        features = {}

        # --- 基础特征 ---
        home_elo = self.elo_system.get_rating(home_team) if self.elo_system else 1500
        away_elo = self.elo_system.get_rating(away_team) if self.elo_system else 1500
        features['elo_diff'] = home_elo - away_elo + 24

        # 赔率
        market_prob = 0.5
        if fetch_odds:
            try:
                odds_df = fetch_odds(self.odds_api_key, game_date)
                if not odds_df.empty:
                    match_odds = odds_df[(odds_df['home_team'] == home_team) & (odds_df['away_team'] == away_team)]
                    if not match_odds.empty:
                        market_prob = match_odds.iloc[0].get('implied_prob', 0.5)
            except:
                pass
        features['market_prob'] = market_prob

        # 先发投手
        sp_home, sp_away = {}, {}
        if fetch_probable_pitchers:
            try:
                pitchers_df = fetch_probable_pitchers(game_date)
                if not pitchers_df.empty:
                    sp_home_row = pitchers_df[(pitchers_df['team'] == home_team) & (pitchers_df['role'] == 'starter')]
                    sp_away_row = pitchers_df[(pitchers_df['team'] == away_team) & (pitchers_df['role'] == 'starter')]
                    if not sp_home_row.empty: sp_home = sp_home_row.iloc[0].to_dict()
                    if not sp_away_row.empty: sp_away = sp_away_row.iloc[0].to_dict()
            except:
                pass
        features['sp_era_diff'] = sp_home.get('era', 4.5) - sp_away.get('era', 4.5)
        features['sp_fip_diff'] = sp_home.get('fip', 4.5) - sp_away.get('fip', 4.5)
        features['sp_csw_diff'] = sp_home.get('csw_pct', 0.28) - sp_away.get('csw_pct', 0.28)
        features['stuff_plus_diff'] = sp_home.get('stuff_plus', 100) - sp_away.get('stuff_plus', 100)

        # 牛棚
        home_bull_ip, away_bull_ip = 4.0, 4.0
        if fetch_bullpen_stats:
            try:
                bullpen_df = fetch_bullpen_stats(game_date)
                if not bullpen_df.empty:
                    home_bull = bullpen_df[bullpen_df['team'] == home_team]
                    away_bull = bullpen_df[bullpen_df['team'] == away_team]
                    if not home_bull.empty: home_bull_ip = home_bull.iloc[0].get('ip_last3', 4.0)
                    if not away_bull.empty: away_bull_ip = away_bull.iloc[0].get('ip_last3', 4.0)
            except:
                pass
        features['bullpen_ip_diff'] = home_bull_ip - away_bull_ip

        features['rest_diff'] = game.get('home_rest', 1) - game.get('away_rest', 1)

        # 天气 & 公园
        weather = {}
        if fetch_openmeteo:
            try:
                weather_df = fetch_openmeteo(game_date)
                if not weather_df.empty:
                    weather = weather_df.iloc[0].to_dict()
            except:
                pass
        temp = weather.get('temp', 70)
        park_factor = get_park_factor(venue) if get_park_factor else 1.0
        features['dynamic_park_factor'] = park_factor * (1 + 0.001 * (temp - 70))

        # 左右投对位 OPS
        home_ops, away_ops = 0.720, 0.720
        if fetch_platoon_splits:
            try:
                platoon_df = fetch_platoon_splits(2026)
                if not platoon_df.empty:
                    home_platoon = platoon_df[platoon_df['team'] == home_team]
                    away_platoon = platoon_df[platoon_df['team'] == away_team]
                    if not home_platoon.empty: home_ops = home_platoon.iloc[0].get('ops', 0.720)
                    if not away_platoon.empty: away_ops = away_platoon.iloc[0].get('ops', 0.720)
            except:
                pass
        features['platoon_ops_diff'] = home_ops - away_ops

        # Statcast 团队数据
        home_stat, away_stat = pd.Series(), pd.Series()
        if fetch_savant_statcast:
            try:
                statcast_df = fetch_savant_statcast(game_date)
                if not statcast_df.empty:
                    home_stat = statcast_df[statcast_df['team'] == home_team].mean()
                    away_stat = statcast_df[statcast_df['team'] == away_team].mean()
            except:
                pass
        for metric in ['launch_speed', 'barrel_pct', 'hard_hit_pct', 'xwoba']:
            features[f'statcast_{metric}_diff'] = home_stat.get(metric, 0) - away_stat.get(metric, 0)

        features['timezone_diff'] = game.get('timezone_diff', 0)
        features['is_day_game'] = 1 if game.get('is_day_game') else 0
        features['home_back2back'] = 1 if game.get('home_back2back') else 0
        features['away_back2back'] = 1 if game.get('away_back2back') else 0

        # 捕手
        if get_catcher_effect:
            catcher_home = get_catcher_effect(home_team)
            catcher_away = get_catcher_effect(away_team)
            features['catcher_era_diff'] = catcher_home.get('era_diff', 0) - catcher_away.get('era_diff', 0)
            features['cs_diff'] = catcher_home.get('cs_diff', 0) - catcher_away.get('cs_diff', 0)
        else:
            features['catcher_era_diff'] = 0
            features['cs_diff'] = 0

        wind_speed = weather.get('wind_speed', 0)
        wind_dir = weather.get('wind_dir', 0)
        features['wind_effect'] = wind_speed * math.cos(math.radians(wind_dir))
        features['temp_effect'] = max(0, temp - 70)
        features['precip_effect'] = 1 if weather.get('precip_prob', 0) > 50 else 0

        # 伤病
        injury_home, injury_away = 0, 0
        if fetch_injuries:
            try:
                injuries_df = fetch_injuries(game_date)
                if not injuries_df.empty:
                    injury_home = injuries_df[injuries_df['team'] == home_team].get('severity', 0).sum()
                    injury_away = injuries_df[injuries_df['team'] == away_team].get('severity', 0).sum()
            except:
                pass
        features['injury_diff'] = injury_home - injury_away

        features['dynamic_pythag_diff'] = game.get('home_pythag', 0.5) - game.get('away_pythag', 0.5)
        features['log5_prob'] = self._log5_prob(home_team, away_team)

        # 滞后特征
        if get_lag_features:
            lag = get_lag_features(home_team, away_team, game_date)
            features['lag30_winrate_diff'] = lag.get('winrate_diff', 0)
            features['lag30_runs_diff'] = lag.get('runs_diff', 0)
        else:
            features['lag30_winrate_diff'] = 0
            features['lag30_runs_diff'] = 0

        features['pitch_movement_diff'] = sp_home.get('movement_plus', 0) - sp_away.get('movement_plus', 0)
        features['k_pct_diff'] = home_stat.get('k_pct', 0.22) - away_stat.get('k_pct', 0.22)
        features['bb_pct_diff'] = home_stat.get('bb_pct', 0.08) - away_stat.get('bb_pct', 0.08)
        features['avg_bat_speed_diff'] = home_stat.get('bat_speed', 0) - away_stat.get('bat_speed', 0)

        features['pitcher_rating_diff'] = (
            get_pitcher_rating(sp_home.get('pitcher_id')) - get_pitcher_rating(sp_away.get('pitcher_id'))
        ) if get_pitcher_rating else 0

        # 赔率变化
        features['odds_change'] = 0
        features['odds_momentum'] = 0
        if fetch_odds:
            try:
                odds_df = fetch_odds(self.odds_api_key, game_date)
                if not odds_df.empty:
                    match = odds_df[(odds_df['home_team'] == home_team) & (odds_df['away_team'] == away_team)]
                    if not match.empty:
                        features['odds_change'] = match.iloc[0].get('change_from_prev', 0)
                        features['odds_momentum'] = match.iloc[0].get('momentum', 0)
            except:
                pass

        # 裁判
        umpire = {}
        if fetch_umpire_data:
            try:
                umpire_df = fetch_umpire_data(game_date)
                if not umpire_df.empty:
                    umpire = umpire_df.iloc[0].to_dict()
            except:
                pass
        features['zone_size'] = umpire.get('zone_size', 1.0)
        features['k_rate'] = umpire.get('k_rate', 0.22)

        # 牛棚可用性
        features['bullpen_availability_diff'] = (
            get_bullpen_availability(home_team) - get_bullpen_availability(away_team)
        ) if get_bullpen_availability else 0

        # ELO 动量
        if get_elo_momentum:
            features['elo_momentum_7d'] = get_elo_momentum(home_team, 7) - get_elo_momentum(away_team, 7)
            features['elo_momentum_30d'] = get_elo_momentum(home_team, 30) - get_elo_momentum(away_team, 30)
        else:
            features['elo_momentum_7d'] = 0
            features['elo_momentum_30d'] = 0

        features['barrel_pa_diff'] = home_stat.get('barrel_per_pa', 0) - away_stat.get('barrel_per_pa', 0)
        features['hardhit_pa_diff'] = home_stat.get('hard_hit_per_pa', 0) - away_stat.get('hard_hit_per_pa', 0)
        features['swing_miss_diff'] = home_stat.get('whiff_pct', 0) - away_stat.get('whiff_pct', 0)
        features['csw_diff'] = sp_home.get('csw_pct', 0.28) - sp_away.get('csw_pct', 0.28)
        features['barrel_bb_pct_diff'] = sp_home.get('barrel_bb_pct', 0.06) - sp_away.get('barrel_bb_pct', 0.06)
        features['sprint_speed_diff'] = home_stat.get('sprint_speed', 27) - away_stat.get('sprint_speed', 27)

        features['pitch_type_matchup_score'] = (
            get_pitch_type_matchup_score(sp_home.get('pitcher_id'), away_team) -
            get_pitch_type_matchup_score(sp_away.get('pitcher_id'), home_team)
        ) if get_pitch_type_matchup_score else 0

        features['home_top3_woba'] = game.get('home_top3_woba', 0.320)
        features['away_top3_woba'] = game.get('away_top3_woba', 0.320)

        features['bt_strength_diff'] = (
            bradley_terry_strength(home_team) - bradley_terry_strength(away_team)
        ) if bradley_terry_strength else 0

        # ========== 新增特征（由 config 控制）==========
        # Glicko2
        if config.RATINGS_ENGINE == 'glicko2':
            league = self.get_glicko_league()
            if league:
                diff, rd_sum = league.get_rating_diff(home_team, away_team)
                features['elo_diff'] = diff
                features['glicko_rd_sum'] = rd_sum

        # 球种对位
        if config.FEATURE_USE_PITCH_MATCHUP and add_matchup_features is not None:
            lookup = self.get_matchup_lookup()
            if lookup is not None and sp_home.get('pitcher_id') and sp_away.get('pitcher_id'):
                features = add_matchup_features(
                    features,
                    sp_home.get('pitcher_id'),
                    sp_away.get('pitcher_id'),
                    game.get('home_top3_ids', []),
                    game.get('away_top3_ids', []),
                    lookup
                )

        # 盘口曲线
        if config.ODDS_USE_CURVE_FEATURES and extract_odds_curve_features is not None:
            trend, vol, rev = extract_odds_curve_features(game_id)
            features['home_odds_trend'] = trend
            features['home_odds_volatility'] = vol
            features['home_odds_reversals'] = rev

        return features

    def _log5_prob(self, home, away):
        return 0.55


# 保持模块级函数兼容
def build_game_features(game):
    model = UnifiedSportsModel()
    return model.build_game_features(game)
