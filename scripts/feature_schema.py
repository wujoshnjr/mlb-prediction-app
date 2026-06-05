# scripts/feature_schema.py
"""Single source of truth for feature order and feature governance.

EXPECTED_FEATURES:
    All runtime / snapshot columns that prediction.py may emit.

MODEL_FEATURES:
    Features allowed into the current main production training pipeline.

TRACKING_ONLY_FEATURES:
    Features kept in prediction reports and snapshots, but excluded from the
    main model until availability, stability, walk-forward contribution and CLV
    contribution are proven.

AVAILABILITY_FLAG_FEATURES:
    Explicit indicators that separate "source unavailable" from "true zero".
"""

from __future__ import annotations

CORE_MODEL_FEATURES = [
    "elo_diff",
    "bt_strength_diff",
    "sp_era_diff",
    "sp_fip_diff",
    "sp_csw_diff",
    "sp_stuff_plus_diff",
    "k_pct_diff",
    "bb_pct_diff",
    "pitcher_rating_diff",
    "bullpen_ip_diff",
    "bullpen_availability_diff",
    "dynamic_park_factor",
    "winrate_diff",
    "timezone_diff",
]

AVAILABILITY_FLAG_FEATURES = [
    "sp_fip_diff_available",
    "sp_csw_diff_available",
    "sp_stuff_plus_diff_available",
    "pitcher_advanced_available",
    "bullpen_context_available",
    "statcast_woba_available",
    "top3_woba_available",
    "weather_available",
    "team_form_available",
    "lineup_context_available",
    "starter_context_available",
    "odds_available",
]

MODEL_FEATURES = CORE_MODEL_FEATURES + AVAILABILITY_FLAG_FEATURES

TRACKING_ONLY_FEATURES = [
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
    "rest_diff",
    "pitch_movement_diff",
    "avg_bat_speed_diff",
    "odds_change",
    "zone_size",
    "k_rate",
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
    "statcast_launch_speed_diff",
    "statcast_barrel_diff",
    "statcast_hard_hit_diff",
    "statcast_woba_diff",
    "platoon_ops_diff",
]

EXPECTED_FEATURES = MODEL_FEATURES + [
    feature for feature in TRACKING_ONLY_FEATURES if feature not in MODEL_FEATURES
]

FEATURE_GRADE_RULES = {
    "A": {
        "availability_rate_min": 0.90,
        "non_zero_rate_min": 0.50,
        "training_allowed": True,
    },
    "B": {
        "availability_rate_min": 0.70,
        "non_zero_rate_min": 0.20,
        "training_allowed": True,
    },
    "C": {
        "availability_rate_min": 0.20,
        "non_zero_rate_min": 0.01,
        "training_allowed": False,
    },
    "D": {
        "availability_rate_min": 0.0,
        "non_zero_rate_min": 0.0,
        "training_allowed": False,
    },
}

FEATURE_GOVERNANCE_NOTES = {
    "CORE_MODEL_FEATURES": "Current main model feature set.",
    "AVAILABILITY_FLAG_FEATURES": "Availability indicators for important sources.",
    "TRACKING_ONLY_FEATURES": "Tracked but excluded from the main model.",
}
