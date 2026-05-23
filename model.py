# model.py
"""
数据聚合模型：统一调用所有数据客户端，生成比赛特征字典。
新功能：支持 Glicko2、球种对位、盘口曲线特征。
"""

import os
import math
import logging
from datetime import datetime
import pandas as pd
import numpy as np

import config

# 导入各类客户端（根据实际项目路径调整）
from scripts.mlb_stats_client import get_schedule, get_probable_pitchers, get_game_info
from scripts.savant_client import get_statcast_team_data
from scripts.odds_client import get_odds, extract_odds_curve_features
from scripts.openmeteo_client import get_weather
from scripts.pitcher_client import get_pitcher_stats
from scripts.bullpen_client import get_bullpen_usage
from scripts.platoon_client import get_platoon_splits
from scripts.catcher_utils import get_catcher_effect
from scripts.park_factors import get_park_factor
from scripts.elo import EloSystem  # 旧的 ELO 模块
from scripts.elo_momentum import get_elo_momentum
from scripts.lag_features import get_lag_features
from scripts.umpire_client import get_umpire_tendency
from scripts.injury_client import get_injury_impact
from scripts.pitch_type_matchup import get_pitch_type_matchup_score
from scripts.bradley_terry import bradley_terry_strength
from scripts.bullpen_availability import get_bullpen_availability
from scripts.player_ratings import get_pitcher_rating

# Glicko2 相关
if config.RATINGS_ENGINE == 'glicko2':
    from scripts.glicko2_ratings import Glicko2League
    from scripts.rating_updater import load_glicko2_league

# Pitch Matchup
if config.FEATURE_USE_PITCH_MATCHUP:
    from scripts.batter_vs_pitch_client import add_matchup_features, build_batter_vs_pitch_lookup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局缓存（按需实现）
_elo_system = EloSystem()
_glicko_league = None
_matchup_lookup = None

def get_glicko_league():
    global _glicko_league
    if _glicko_league is None:
        _glicko_league = load_glicko2_league()
    return _glicko_league

def get_matchup_lookup():
    global _matchup_lookup
    if _matchup_lookup is None and config.FEATURE_USE_PITCH_MATCHUP:
        # 尝试从文件加载，若没有则构建（可能需要离线任务）
        lookup_path = 'data/batter_vs_pitch_lookup.parquet'
        if os.path.exists(lookup_path):
            _matchup_lookup = pd.read_parquet(lookup_path)
        else:
            logger.warning("Matchup lookup 文件缺失，将使用默认值")
            _matchup_lookup = pd.DataFrame(columns=['batter_id', 'pitch_type', 'woba', 'whiff_rate', 'hard_hit_rate', 'avg_run_value'])
    return _matchup_lookup


def build_game_features(game):
    """
    为单场比赛生成完整特征字典。
    game: 比赛信息 dict，包含 teams, pitchers, datetime, venue 等。
    """
    home_team = game['home_team']
    away_team = game['away_team']
    home_sp = game.get('home_probable_pitcher_id')
    away_sp = game.get('away_probable_pitcher_id')
    game_date = game['date']
    venue = game.get('venue')

    features = {}

    # --------------------- 基础特征（保持不变）---------------------
    # ELO 相关
    home_elo = _elo_system.get_rating(home_team)
    away_elo = _elo_system.get_rating(away_team)
    features['elo_diff'] = home_elo - away_elo + 24  # 主场优势

    # 赔率隐含概率
    odds = get_odds(game)
    market_prob = odds.get('home_implied_prob', 0.5)
    features['market_prob'] = market_prob

    # 先发投手
    sp_home = get_pitcher_stats(home_sp) if home_sp else {}
    sp_away = get_pitcher_stats(away_sp) if away_sp else {}
    features['sp_era_diff'] = sp_home.get('era', 4.5) - sp_away.get('era', 4.5)
    features['sp_fip_diff'] = sp_home.get('fip', 4.5) - sp_away.get('fip', 4.5)
    features['sp_csw_diff'] = sp_home.get('csw_pct', 0.28) - sp_away.get('csw_pct', 0.28)
    features['stuff_plus_diff'] = sp_home.get('stuff_plus', 100) - sp_away.get('stuff_plus', 100)

    # 牛棚
    bull_home = get_bullpen_usage(home_team)
    bull_away = get_bullpen_usage(away_team)
    features['bullpen_ip_diff'] = bull_home.get('ip_last3', 4.0) - bull_away.get('ip_last3', 4.0)

    # 休息天数
    features['rest_diff'] = game.get('home_rest', 1) - game.get('away_rest', 1)

    # 动态公园因子
    weather = get_weather(venue, game_date)
    temperature = weather.get('temp', 70)
    park_factor = get_park_factor(venue)
    features['dynamic_park_factor'] = park_factor * (1 + 0.001 * (temperature - 70))

    # 左右投对位 OPS
    platoon_home = get_platoon_splits(home_team, away_sp_handedness(sp_away))
    platoon_away = get_platoon_splits(away_team, home_sp_handedness(sp_home))
    features['platoon_ops_diff'] = platoon_home.get('ops', 0.720) - platoon_away.get('ops', 0.720)

    # Statcast 团队打击
    stat_home = get_statcast_team_data(home_team)
    stat_away = get_statcast_team_data(away_team)
    for metric in ['launch_speed', 'barrel_pct', 'hard_hit_pct', 'xwoba']:
        features[f'statcast_{metric}_diff'] = stat_home.get(metric, 0) - stat_away.get(metric, 0)

    # 时区 / 日场
    features['timezone_diff'] = game.get('timezone_diff', 0)
    features['is_day_game'] = 1 if game.get('is_day_game') else 0

    # 背靠背
    features['home_back2back'] = 1 if game.get('home_back2back') else 0
    features['away_back2back'] = 1 if game.get('away_back2back') else 0

    # 捕手
    catcher_effect = get_catcher_effect(home_team)
    features['catcher_era_diff'] = catcher_effect.get('era_diff', 0)
    features['cs_diff'] = catcher_effect.get('cs_diff', 0)

    # 天气
    wind_speed = weather.get('wind_speed', 0)
    wind_dir = weather.get('wind_dir', 0)
    features['wind_effect'] = wind_speed * math.cos(math.radians(wind_dir))  # 简化
    features['temp_effect'] = max(0, temperature - 70)
    features['precip_effect'] = 1 if weather.get('precip_prob', 0) > 50 else 0

    # 伤病
    injury_diff = get_injury_impact(home_team) - get_injury_impact(away_team)
    features['injury_diff'] = injury_diff

    # Pythag 胜率
    # 假设已有球队得分/失分数据
    features['dynamic_pythag_diff'] = game.get('home_pythag', 0.5) - game.get('away_pythag', 0.5)

    # Log5
    features['log5_prob'] = log5_probability(home_team, away_team)

    # 滞后特征
    lag = get_lag_features(home_team, away_team, game_date)
    features['lag30_winrate_diff'] = lag.get('winrate_diff', 0)
    features['lag30_runs_diff'] = lag.get('runs_diff', 0)

    # 投球位移 / 挥空率等
    features['pitch_movement_diff'] = sp_home.get('movement_plus', 0) - sp_away.get('movement_plus', 0)
    features['k_pct_diff'] = stat_home.get('k_pct', 0.22) - stat_away.get('k_pct', 0.22)
    features['bb_pct_diff'] = stat_home.get('bb_pct', 0.08) - stat_away.get('bb_pct', 0.08)
    features['avg_bat_speed_diff'] = stat_home.get('bat_speed', 0) - stat_away.get('bat_speed', 0)

    # 投手综合评分
    features['pitcher_rating_diff'] = get_pitcher_rating(home_sp) - get_pitcher_rating(away_sp)

    # 盘口变化
    features['odds_change'] = odds.get('change_from_prev', 0)
    features['odds_momentum'] = odds.get('momentum', 0)

    # 裁判
    ump = get_umpire_tendency(game.get('umpire_id'))
    features['zone_size'] = ump.get('zone_size', 1.0)
    features['k_rate'] = ump.get('k_rate', 0.22)

    # 牛棚可用性
    features['bullpen_availability_diff'] = get_bullpen_availability(home_team) - get_bullpen_availability(away_team)

    # ELO 动量
    features['elo_momentum_7d'] = get_elo_momentum(home_team, 7) - get_elo_momentum(away_team, 7)
    features['elo_momentum_30d'] = get_elo_momentum(home_team, 30) - get_elo_momentum(away_team, 30)

    # 更多 Statcast
    features['barrel_pa_diff'] = stat_home.get('barrel_per_pa', 0) - stat_away.get('barrel_per_pa', 0)
    features['hardhit_pa_diff'] = stat_home.get('hard_hit_per_pa', 0) - stat_away.get('hard_hit_per_pa', 0)
    features['swing_miss_diff'] = stat_home.get('whiff_pct', 0) - stat_away.get('whiff_pct', 0)
    features['csw_diff'] = sp_home.get('csw_pct', 0.28) - sp_away.get('csw_pct', 0.28)
    features['barrel_bb_pct_diff'] = sp_home.get('barrel_bb_pct', 0.06) - sp_away.get('barrel_bb_pct', 0.06)
    features['sprint_speed_diff'] = stat_home.get('sprint_speed', 27) - stat_away.get('sprint_speed', 27)

    # 球种对位分数（原有简化版）
    features['pitch_type_matchup_score'] = get_pitch_type_matchup_score(home_sp, away_team) - get_pitch_type_matchup_score(away_sp, home_team)

    # 前三棒 wOBA
    features['home_top3_woba'] = game.get('home_top3_woba', 0.320)
    features['away_top3_woba'] = game.get('away_top3_woba', 0.320)

    # Bradley-Terry
    features['bt_strength_diff'] = bradley_terry_strength(home_team) - bradley_terry_strength(away_team)

    # ===================== 新特征（由 config 控制） =====================
    # 1. Glicko2 替代 ELO
    if config.RATINGS_ENGINE == 'glicko2':
        league = get_glicko_league()
        diff, rd_sum = league.get_rating_diff(home_team, away_team)
        # 替换 elo_diff 和 elo_momentum 相关特征（或者追加新特征，取决于你想保留旧特征对比）
        features['elo_diff'] = diff  # 覆盖或新键
        features['glicko_rd_sum'] = rd_sum
        # 注意：elo_momentum 此时不适用，可置零或删除，这里保留原有但可能失真
        # 更好的做法是另加 glicko_momentum，但现在简化处理

    # 2. 球种对位特征
    if config.FEATURE_USE_PITCH_MATCHUP:
        lookup = get_matchup_lookup()
        home_top3 = game.get('home_top3_ids', [])
        away_top3 = game.get('away_top3_ids', [])
        features = add_matchup_features(features, home_sp, away_sp, home_top3, away_top3, lookup)

    # 3. 盘口曲线特征
    if config.ODDS_USE_CURVE_FEATURES:
        game_id = game.get('game_id')
        trend, vol, rev = extract_odds_curve_features(game_id)
        features['home_odds_trend'] = trend
        features['home_odds_volatility'] = vol
        features['home_odds_reversals'] = rev

    return features


# 辅助函数
def home_sp_handedness(sp_dict):
    return sp_dict.get('throws', 'R')

def away_sp_handedness(sp_dict):
    return sp_dict.get('throws', 'R')

def log5_probability(team_a, team_b):
    # 简单实现，可替换为从数据获取
    return 0.55
