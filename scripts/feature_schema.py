# scripts/feature_schema.py
"""Single source of truth for ML feature order.

Only features currently produced by prediction.py and suitable for training are
listed here. Experimental odds-curve and pitch-usage features remain disabled
until their data pipeline is validated and should not enter training as
permanently-zero columns.
"""

EXPECTED_FEATURES = [
    "elo_diff",
    "sp_era_diff",
    "sp_fip_diff",
    "sp_stuff_plus_diff",
    "sp_csw_diff",
    "bullpen_ip_diff",
    "rest_diff",
    "dynamic_park_factor",
    "platoon_ops_diff",
    "statcast_launch_speed_diff",
    "statcast_barrel_diff",
    "statcast_hard_hit_diff",
    "statcast_woba_diff",
    "timezone_diff",
    "is_day_game",
    "back2back_diff",
    "catcher_era_diff",
    "cs_diff",
    "wind_effect",
    "temp_effect",
    "precip_effect",
    "injury_diff",
    "dynamic_pythag_diff",
    "log5_prob",
    "lag30_winrate_diff",
    "lag30_runs_diff",
    "pitch_movement_diff",
    "k_pct_diff",
    "bb_pct_diff",
    "avg_bat_speed_diff",
    "pitcher_rating_diff",
    "odds_change",
    "zone_size",
    "k_rate",
    "bullpen_availability_diff",
    "elo_momentum_7d",
    "elo_momentum_30d",
    "barrel_pa_diff",
    "hardhit_pa_diff",
    "swing_miss_diff",
    "csw_diff",
    "barrel_bb_pct_diff",
    "sprint_speed_diff",
    "pitch_type_matchup_score",
    "top3_woba_diff",
    "winrate_diff",
    "bt_strength_diff",
]
