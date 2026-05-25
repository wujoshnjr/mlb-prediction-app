# config.py
"""Runtime feature and model configuration.

Keep experimental features disabled until their data sources and validation
checks are confirmed healthy in GitHub Actions.
"""

# Rating engine used by prediction.py. Keep Elo active until the rating rebuild
# has generated and validated clean Elo/Glicko2 state files.
RATINGS_ENGINE = "elo"  # "elo" or "glicko2"

# Ensemble/meta-model options
MODEL_META = "lr"  # "lr" or "elasticnet"
MODEL_USE_MLP = False

# Backtest mode
WALKFORWARD_STRICT = False

# Feature gates. Do not turn these on until the corresponding pipeline is
# producing non-empty, validated features.
FEATURE_USE_PITCH_MATCHUP = False
FEATURE_USE_PITCH_USAGE = False
NRFI_USE_ML = False
ODDS_USE_CURVE_FEATURES = False
