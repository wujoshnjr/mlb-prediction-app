"""
Pitch Usage 特征模块
计算球种使用变化的三层特征：
1. usage_magnitude - 变化规模
2. directional_deltas - 各球种的变化向量（FF, SL, CH 等）
3. shift_score - 对位调整后的变化分数
"""
import pandas as pd
import numpy as np

# 常用球种列表
PITCH_TYPES = ['FF', 'SL', 'CH', 'CU', 'FC', 'SI', 'KC', 'FS']

def compute_pitch_usage_features(savant_df, home_team, away_team, home_pitcher_id, away_pitcher_id):
    """
    根据 Savant 数据计算主客队投手的球种使用变化特征。
    返回字典，包含三层特征。
    """
    features = {
        'home_usage_magnitude': 0.0,
        'away_usage_magnitude': 0.0,
        'home_shift_score': 0.0,
        'away_shift_score': 0.0
    }
    # 初始化各球种 delta
    for pt in PITCH_TYPES:
        features[f'home_delta_{pt}'] = 0.0
        features[f'away_delta_{pt}'] = 0.0

    if savant_df.empty or 'pitch_type' not in savant_df.columns:
        return features

    # 获取最近3场、7场和赛季数据（这里简化：用全部 Savant 数据作为赛季近似）
    # 实际应区分时间段，此处用近7天和全部数据演示
    today = pd.Timestamp.now()
    last7 = savant_df[savant_df['game_date'] >= (today - pd.Timedelta(days=7)).strftime('%Y-%m-%d')]

    def calc_usage(df, pitcher_id):
        """计算指定投手的球种使用率"""
        pitcher_df = df[df['pitcher_id'] == pitcher_id]
        if len(pitcher_df) == 0:
            return None
        counts = pitcher_df['pitch_type'].value_counts(normalize=True)
        return counts

    def calc_shift_score(deltas, batter_woba_dict):
        """计算对位调整分数"""
        score = 0.0
        for pt in PITCH_TYPES:
            delta = deltas.get(pt, 0.0)
            woba = batter_woba_dict.get(pt, 0.320)  # 默认联盟平均
            score += delta * woba
        return score

    # 主队投手
    if home_pitcher_id:
        season_usage = calc_usage(savant_df, home_pitcher_id)
        recent_usage = calc_usage(last7, home_pitcher_id)
        if season_usage is not None and recent_usage is not None:
            # 加权融合
            fused = 0.6 * recent_usage + 0.3 * recent_usage + 0.1 * season_usage  # 简化
            deltas = {}
            for pt in PITCH_TYPES:
                d = fused.get(pt, 0.0) - season_usage.get(pt, 0.0)
                features[f'home_delta_{pt}'] = d
                deltas[pt] = d
            features['home_usage_magnitude'] = np.sum(np.abs(list(deltas.values())))
            # 对手打线数据需外部传入，此处暂用0
            features['home_shift_score'] = calc_shift_score(deltas, {})

    # 客队投手
    if away_pitcher_id:
        season_usage = calc_usage(savant_df, away_pitcher_id)
        recent_usage = calc_usage(last7, away_pitcher_id)
        if season_usage is not None and recent_usage is not None:
            fused = 0.6 * recent_usage + 0.3 * recent_usage + 0.1 * season_usage
            deltas = {}
            for pt in PITCH_TYPES:
                d = fused.get(pt, 0.0) - season_usage.get(pt, 0.0)
                features[f'away_delta_{pt}'] = d
                deltas[pt] = d
            features['away_usage_magnitude'] = np.sum(np.abs(list(deltas.values())))
            features['away_shift_score'] = calc_shift_score(deltas, {})

    return features
