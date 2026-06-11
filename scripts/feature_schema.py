# scripts/feature_schema.py
"""Single source of truth for feature order and feature governance.

MODEL_FEATURES:
    Current main baseline model feature set.

TRACKING_ONLY_FEATURES:
    Features stored in snapshots and reports but excluded from the main model.

CANDIDATE_SHADOW_FEATURES:
    Features allowed only in shadow model lab experiments. These must not enter
    train_ensemble.py directly. Promotion must go through feature promotion reports,
    walk-forward evidence, calibration checks, and governance review.

Do not bypass this file to promote features.
"""

from __future__ import annotations


CORE_MODEL_FEATURES = [
    "elo_diff",
    "bt_strength_diff",
    "sp_era_diff",
    "pitcher_rating_diff",
    "dynamic_park_factor",
    "winrate_diff",
    "timezone_diff",
]

DEFERRED_ZERO_MODEL_FEATURES = [
    "sp_fip_diff",
    "sp_csw_diff",
    "sp_stuff_plus_diff",
    "k_pct_diff",
    "bb_pct_diff",
    "bullpen_ip_diff",
    "bullpen_availability_diff",
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

TRACKING_ONLY_FEATURES = DEFERRED_ZERO_MODEL_FEATURES + [
    "is_day_game",
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
    "back2back_diff",
    "games_last_3d_diff",
    "games_last_7d_diff",
    "rest_pressure_diff",
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

CANDIDATE_SHADOW_FEATURES = DEFERRED_ZERO_MODEL_FEATURES + [
    "statcast_woba_diff",
    "top3_woba_diff",
    "barrel_pa_diff",
    "hardhit_pa_diff",
    "avg_bat_speed_diff",
    "swing_miss_diff",
    "pitch_movement_diff",
    "pitch_type_matchup_score",
    "platoon_ops_diff",
    "lag30_winrate_diff",
    "lag30_runs_diff",
    "rest_diff",
    "wind_effect",
    "temp_effect",
    "precip_effect",
    "odds_change",
    "injury_diff",
    "lineup_context_available",
]

EXPECTED_FEATURES = MODEL_FEATURES + [
    feature for feature in TRACKING_ONLY_FEATURES if feature not in MODEL_FEATURES
]

FEATURE_GROUPS = {
    "core_strength": [
        "elo_diff",
        "bt_strength_diff",
        "winrate_diff",
        "dynamic_pythag_diff",
        "log5_prob",
        "elo_momentum_7d",
        "elo_momentum_30d",
    ],
    "starting_pitcher": [
        "sp_era_diff",
        "sp_fip_diff",
        "sp_csw_diff",
        "sp_stuff_plus_diff",
        "k_pct_diff",
        "bb_pct_diff",
        "pitcher_rating_diff",
        "pitch_movement_diff",
        "pitch_type_matchup_score",
    ],
    "bullpen": [
        "bullpen_ip_diff",
        "bullpen_availability_diff",
    ],
    "park_weather": [
        "dynamic_park_factor",
        "wind_effect",
        "temp_effect",
        "precip_effect",
        "is_day_game",
    ],
    "lineup": [
        "lineup_context_available",
        "catcher_era_diff",
        "cs_diff",
        "top3_woba_diff",
        "platoon_ops_diff",
    ],
    "statcast_batting": [
        "statcast_woba_diff",
        "barrel_pa_diff",
        "hardhit_pa_diff",
        "avg_bat_speed_diff",
        "swing_miss_diff",
        "barrel_bb_pct_diff",
        "sprint_speed_diff",
        "statcast_launch_speed_diff",
        "statcast_barrel_diff",
        "statcast_hard_hit_diff",
    ],
    "market": [
        "odds_change",
    ],
    "rest_travel": [
        "timezone_diff",
        "lag30_winrate_diff",
        "lag30_runs_diff",
        "rest_diff",
        "back2back_diff",
        "games_last_3d_diff",
        "games_last_7d_diff",
        "rest_pressure_diff",
    ],
    "injuries": [
        "injury_diff",
    ],
    "availability": AVAILABILITY_FLAG_FEATURES,
}


def _group_for_feature(feature: str) -> str:
    for group, features in FEATURE_GROUPS.items():
        if feature in features:
            return group
    return "ungrouped"


def _availability_required(feature: str) -> bool:
    return feature in AVAILABILITY_FLAG_FEATURES or feature.endswith("_available")


def _leakage_risk(feature: str) -> str:
    if feature in {
        "home_win",
        "home_score",
        "away_score",
        "final_score",
        "settled_result",
        "actual_winner",
        "actual_result",
    }:
        return "high"

    if feature in {"odds_change"}:
        return "medium"

    return "low"


FEATURE_METADATA = {}

for feature in EXPECTED_FEATURES:
    FEATURE_METADATA[feature] = {
        "group": _group_for_feature(feature),
        "leakage_risk": _leakage_risk(feature),
        "availability_required": _availability_required(feature),
        "allow_in_main_model": feature in MODEL_FEATURES,
        "allow_in_shadow_model": feature in MODEL_FEATURES or feature in CANDIDATE_SHADOW_FEATURES,
        "description": (
            "Current main baseline feature."
            if feature in MODEL_FEATURES
            else "Tracked feature; requires promotion evidence before model use."
        ),
    }

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
    "MODEL_FEATURES": "Only features allowed into the current main training pipeline.",
    "TRACKING_ONLY_FEATURES": "Tracked but excluded from the main model.",
    "CANDIDATE_SHADOW_FEATURES": "May be used only by shadow model lab and feature promotion research.",
    "FEATURE_METADATA": "Governance metadata for feature review, leakage risk and promotion controls.",
}
