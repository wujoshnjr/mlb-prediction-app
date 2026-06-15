# scripts/feature_schema.py
"""Single source of truth for feature order and feature governance.

CORE_MODEL_FEATURES:
    Only features allowed into active model training and active serving.

DEFERRED_ZERO_MODEL_FEATURES:
    Tracked features currently excluded from active model because their
    non-zero evidence or availability is not yet strong enough.

SHADOW_CANDIDATE_FEATURES:
    Features allowed only for shadow research and diagnostic reports. These must
    not enter train_ensemble.py or prediction.py active artifact serving without
    feature promotion evidence and governance review.

Do not bypass this file to promote features.
"""

from __future__ import annotations

import builtins
import hashlib
import json
from typing import Any


MODEL_FEATURE_VERSION = "core_v4_canonical_training_samples"


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

# Backward-compatibility shim for prediction.py. The current prediction runtime
# references AVAILABILITY_FLAG_FEATURES as an unqualified global inside
# apply_feature_availability_flags(). Because prediction.py imports selected
# symbols from this module instead of importing AVAILABILITY_FLAG_FEATURES
# directly, expose the list through builtins so runtime generation cannot crash.
# This does not promote availability flags into the active model feature set.
builtins.AVAILABILITY_FLAG_FEATURES = list(AVAILABILITY_FLAG_FEATURES)


MODEL_FEATURES = list(CORE_MODEL_FEATURES)


TRACKING_ONLY_FEATURES = DEFERRED_ZERO_MODEL_FEATURES + AVAILABILITY_FLAG_FEATURES + [
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


SHADOW_CANDIDATE_FEATURES = [
    "sp_fip_diff",
    "sp_csw_diff",
    "sp_stuff_plus_diff",
    "k_pct_diff",
    "bb_pct_diff",
    "bullpen_ip_diff",
    "bullpen_availability_diff",
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

# Legacy name used by scripts/feature_promotion_report.py and older tests.
# Keep it as an alias, not a separate source of truth.
CANDIDATE_SHADOW_FEATURES = list(SHADOW_CANDIDATE_FEATURES)


EXPECTED_FEATURES = CORE_MODEL_FEATURES + [
    feature for feature in TRACKING_ONLY_FEATURES if feature not in CORE_MODEL_FEATURES
]


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        output.append(value)

    return output


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)

    return sorted(duplicates)


def _stable_hash(values: list[str]) -> str:
    payload = json.dumps(
        list(values),
        ensure_ascii=True,
        sort_keys=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


MODEL_FEATURE_SCHEMA_HASH = _stable_hash(CORE_MODEL_FEATURES)


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
        "allow_in_main_model": feature in CORE_MODEL_FEATURES,
        "allow_in_shadow_model": feature in CORE_MODEL_FEATURES
        or feature in SHADOW_CANDIDATE_FEATURES,
        "description": (
            "Current active core model feature."
            if feature in CORE_MODEL_FEATURES
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
    "CORE_MODEL_FEATURES": "Only features allowed into active training and active serving.",
    "DEFERRED_ZERO_MODEL_FEATURES": "Tracked but excluded from active model until evidence improves.",
    "AVAILABILITY_FLAG_FEATURES": "Tracked availability indicators; excluded from active model by default.",
    "MODEL_FEATURES": "Backward-compatible alias for CORE_MODEL_FEATURES.",
    "TRACKING_ONLY_FEATURES": "Tracked but excluded from active model.",
    "SHADOW_CANDIDATE_FEATURES": "May be used only by shadow model lab and feature promotion research.",
    "CANDIDATE_SHADOW_FEATURES": "Backward-compatible alias for SHADOW_CANDIDATE_FEATURES.",
    "FEATURE_METADATA": "Governance metadata for feature review, leakage risk and promotion controls.",
}


def get_core_model_features() -> list[str]:
    return list(CORE_MODEL_FEATURES)


def get_deferred_zero_model_features() -> list[str]:
    return list(DEFERRED_ZERO_MODEL_FEATURES)


def get_shadow_candidate_features() -> list[str]:
    return list(SHADOW_CANDIDATE_FEATURES)


def get_model_feature_schema_hash() -> str:
    return _stable_hash(CORE_MODEL_FEATURES)


def validate_no_overlap() -> dict[str, Any]:
    duplicate_core_features = _duplicates(CORE_MODEL_FEATURES)
    duplicate_deferred_zero_features = _duplicates(DEFERRED_ZERO_MODEL_FEATURES)
    duplicate_shadow_candidate_features = _duplicates(SHADOW_CANDIDATE_FEATURES)

    deferred_overlap_with_core = sorted(
        set(CORE_MODEL_FEATURES).intersection(DEFERRED_ZERO_MODEL_FEATURES)
    )
    shadow_overlap_with_core = sorted(
        set(CORE_MODEL_FEATURES).intersection(SHADOW_CANDIDATE_FEATURES)
    )

    errors: list[str] = []

    if duplicate_core_features:
        errors.append("duplicate_core_features")

    if duplicate_deferred_zero_features:
        errors.append("duplicate_deferred_zero_features")

    if duplicate_shadow_candidate_features:
        errors.append("duplicate_shadow_candidate_features")

    if deferred_overlap_with_core:
        errors.append("deferred_overlap_with_core")

    if shadow_overlap_with_core:
        errors.append("shadow_overlap_with_core")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "duplicate_core_features": duplicate_core_features,
        "duplicate_deferred_zero_features": duplicate_deferred_zero_features,
        "duplicate_shadow_candidate_features": duplicate_shadow_candidate_features,
        "deferred_overlap_with_core": deferred_overlap_with_core,
        "shadow_overlap_with_core": shadow_overlap_with_core,
        "core_feature_count": len(CORE_MODEL_FEATURES),
        "deferred_zero_feature_count": len(DEFERRED_ZERO_MODEL_FEATURES),
        "shadow_candidate_feature_count": len(SHADOW_CANDIDATE_FEATURES),
        "feature_schema_version": MODEL_FEATURE_VERSION,
        "feature_schema_hash": get_model_feature_schema_hash(),
    }
