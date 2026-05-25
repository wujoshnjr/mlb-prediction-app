# scripts/feature_schema.py
"""
共用特徵清單，prediction.py 與 train_ensemble.py 均由此導入，
確保訓練與推論維度一致。
"""
EXPECTED_FEATURES = [
    'elo_diff',
    'sp_era_diff', 'sp_fip_diff', 'sp_stuff_plus_diff', 'sp_csw_diff',
    'bullpen_ip_diff', 'rest_diff',
    'dynamic_park_factor',
    'platoon_ops_diff', 'statcast_launch_speed_diff', 'statcast_barrel_diff',
    'statcast_hard_hit_diff', 'statcast_woba_diff',
    'timezone_diff', 'is_day_game', 'back2back_diff',
    'catcher_era_diff', 'cs_diff', 'wind_effect',
    'temp_effect', 'precip_effect', 'injury_diff',
    'dynamic_pythag_diff', 'log5_prob', 'lag30_winrate_diff', 'lag30_runs_diff',
    'pitch_movement_diff',
    'k_pct_diff', 'bb_pct_diff', 'avg_bat_speed_diff',
    'pitcher_rating_diff', 'odds_change', 'odds_momentum',
    'zone_size', 'k_rate', 'bullpen_availability_diff',
    'elo_momentum_7d', 'elo_momentum_30d', 'barrel_pa_diff', 'hardhit_pa_diff',
    'swing_miss_diff', 'csw_diff', 'barrel_bb_pct_diff',
    'sprint_speed_diff', 'pitch_type_matchup_score',
    'top3_woba_diff', 'winrate_diff', 'bt_strength_diff',
    # Pitch Usage 特征（暂时保留，若未启用数据源则为零）
    'home_usage_magnitude', 'away_usage_magnitude',
    'home_shift_score', 'away_shift_score',
    'home_delta_FF', 'home_delta_SL', 'home_delta_CH', 'home_delta_CU',
    'home_delta_FC', 'home_delta_SI', 'home_delta_KC', 'home_delta_FS',
    'away_delta_FF', 'away_delta_SL', 'away_delta_CH', 'away_delta_CU',
    'away_delta_FC', 'away_delta_SI', 'away_delta_KC', 'away_delta_FS'
]
