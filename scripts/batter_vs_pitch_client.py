# scripts/batter_vs_pitch_client.py
"""
打者-投手对位特征构建
通过 Statcast (Savant) 数据，构建打者面对各球种的历史表现查找表，
并为每场比赛生成投手武器库 vs 对方前段打线的对位优势分数。
"""

import pandas as pd
import numpy as np
import os
import logging
from scripts.savant_client import fetch_statcast_data  # 假设已有这个函数

logger = logging.getLogger(__name__)

# 联盟平均 wOBA 等基线（2025 MLB 赛季）
LEAGUE_AVG_WOBA = 0.320
LEAGUE_AVG_WHIFF = 0.25
LEAGUE_AVG_HARD_HIT = 0.35
LEAGUE_AVG_RUN_VALUE = 0.0

DEFAULT_PERFORMANCE = {
    'woba': LEAGUE_AVG_WOBA,
    'whiff_rate': LEAGUE_AVG_WHIFF,
    'hard_hit_rate': LEAGUE_AVG_HARD_HIT,
    'avg_run_value': LEAGUE_AVG_RUN_VALUE
}


def build_batter_vs_pitch_lookup(start_date: str, end_date: str, save_path: str = None) -> pd.DataFrame:
    """
    从 Savant 数据构建「打者-球种」历史表现表。
    
    参数:
        start_date: 起始日期 'YYYY-MM-DD'
        end_date:   结束日期
        save_path:  可选，保存为 parquet 的路径
    
    返回:
        DataFrame, 列: batter_id, pitch_type, woba, whiff_rate, hard_hit_rate, avg_run_value
    """
    logger.info(f"正在拉取 {start_date} 至 {end_date} 的逐球数据...")
    df = fetch_statcast_data(start_date, end_date)  # 实际需替换为你的 Savant 客户端调用

    required_cols = ['batter', 'pitch_type', 'woba_value', 'description',
                     'launch_speed', 'events']
    df = df[required_cols].dropna(subset=['batter', 'pitch_type'])
    df = df.rename(columns={'batter': 'batter_id'})

    # 定义好球挥空
    swinging_strike = df['description'].isin(['swinging_strike', 'swinging_strike_blocked'])
    df['is_swing'] = df['description'].isin(['swinging_strike', 'swinging_strike_blocked',
                                             'foul', 'hit_into_play'])
    df['is_hard_hit'] = (df['launch_speed'] >= 95).astype(int)

    grouped = df.groupby(['batter_id', 'pitch_type']).agg(
        woba=('woba_value', 'mean'),
        whiff_rate=('is_swing', lambda x: (x & swinging_strike.loc[x.index]).sum() / x.sum() if x.sum() > 0 else np.nan),
        hard_hit_rate=('is_hard_hit', lambda x: x.mean() if x.sum() > 0 else np.nan),
        avg_run_value=('events', lambda x: x.map(_event_to_run_value).mean())
    ).reset_index()

    # 用联盟均值填充缺失
    for col, default in [('woba', LEAGUE_AVG_WOBA), ('whiff_rate', LEAGUE_AVG_WHIFF),
                         ('hard_hit_rate', LEAGUE_AVG_HARD_HIT), ('avg_run_value', LEAGUE_AVG_RUN_VALUE)]:
        grouped[col] = grouped[col].fillna(default)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        grouped.to_parquet(save_path, index=False)
        logger.info(f"对位数据已保存至 {save_path}")

    return grouped


def _event_to_run_value(event: str) -> float:
    """简化的事件运行价值映射（用于平均 run_value）"""
    mapping = {
        'single': 0.47, 'double': 0.77, 'triple': 1.04, 'home_run': 1.40,
        'walk': 0.33, 'hit_by_pitch': 0.34, 'strikeout': -0.30,
        'field_out': -0.27, 'force_out': -0.30, 'grounded_into_double_play': -0.80,
        'sac_fly': -0.20, 'sac_bunt': -0.35
    }
    return mapping.get(event, 0.0)


def get_batter_vs_pitch(batter_id: int, pitch_type: str, lookup: pd.DataFrame) -> dict:
    """
    查询单个打者对特定球种的表现。
    若不存在则返回联盟均值。
    """
    entry = lookup[(lookup['batter_id'] == batter_id) & (lookup['pitch_type'] == pitch_type)]
    if entry.empty:
        return DEFAULT_PERFORMANCE.copy()
    row = entry.iloc[0]
    return {
        'woba': row['woba'],
        'whiff_rate': row['whiff_rate'],
        'hard_hit_rate': row['hard_hit_rate'],
        'avg_run_value': row['avg_run_value']
    }


def get_pitcher_arsenal(pitcher_id: int, start_date: str, end_date: str) -> dict:
    """
    获取投手近期各球种使用比例。
    返回: {'FF': 0.45, 'SL': 0.30, ...}   (值按使用频率归一化)
    """
    try:
        df = fetch_statcast_data(start_date, end_date)
        pitcher_df = df[df['pitcher'] == pitcher_id]
        if pitcher_df.empty:
            return {}
        usage = pitcher_df['pitch_type'].value_counts(normalize=True).to_dict()
        return usage
    except Exception as e:
        logger.warning(f"无法获取投手 {pitcher_id} 武器库: {e}")
        return {}


def compute_matchup_score(pitcher_id: int, batter_ids: list,
                          lookup: pd.DataFrame, lookback_days: int = 60) -> float:
    """
    计算投手对一组打者的对位优势分数 (加权平均 wOBA 差值)。
    值越负代表投手越克制打者。
    
    参数:
        pitcher_id: 投手 ID
        batter_ids: 打者 ID 列表（一般取前 3 棒）
        lookup:      打者-球种查找表
        lookback_days: 投手武器库的回溯天数
    """
    end = pd.Timestamp.today().strftime('%Y-%m-%d')
    start = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    arsenal = get_pitcher_arsenal(pitcher_id, start, end)
    if not arsenal:
        return 0.0  # 无数据，视为中性

    total_usage = sum(arsenal.values())
    score = 0.0
    valid_batters = 0
    for batter in batter_ids:
        batter_score = 0.0
        for pitch, usage in arsenal.items():
            perf = get_batter_vs_pitch(batter, pitch, lookup)
            # 使用 wOBA 与联盟平均的差值作为权重
            batter_score += (usage / total_usage) * (perf['woba'] - LEAGUE_AVG_WOBA)
        if batter_score != 0.0:
            score += batter_score
            valid_batters += 1
    if valid_batters == 0:
        return 0.0
    return score / valid_batters


def add_matchup_features(features: dict, home_pitcher_id: int, away_pitcher_id: int,
                         home_top3: list, away_top3: list,
                         lookup: pd.DataFrame) -> dict:
    """
    向特征字典中追加两个对位特征：
        home_matchup_adv : 主队投手对客队前三棒的优势（负值利于投手）
        away_matchup_adv : 客队投手对主队前三棒的优势
    """
    home_adv = compute_matchup_score(home_pitcher_id, away_top3, lookup)
    away_adv = compute_matchup_score(away_pitcher_id, home_top3, lookup)
    features['home_matchup_adv'] = home_adv
    features['away_matchup_adv'] = away_adv
    return features
