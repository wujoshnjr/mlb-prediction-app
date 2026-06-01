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
# Clean forward-collected prediction snapshot pipeline.
# Legacy historical rows must not be treated as clean training samples.
PIPELINE_VERSION = "baseline_v2_clean"
SNAPSHOT_POLICY = "first_seen_pregame"
BETTING_MODE = "paper_trading"
ALLOW_LEGACY_TRAINING_DATA = False
SNAPSHOT_STORE_FILE = "data/prediction_snapshots.csv"

# Temporary early-model threshold.
# This allows the clean model pipeline to produce an experimental artifact
# for engineering validation before the production 300+ sample threshold.
MIN_CLEAN_TRAIN_SAMPLES = 60

# Moneyline recommendation gate.
# A paper-trading Moneyline recommendation is created only when the model
# probability exceeds the no-vig market probability by at least this amount.
MIN_MONEYLINE_EDGE = 0.03

# Maximum paper-trading stake fraction for one Moneyline recommendation.
# Kelly sizing is capped to avoid oversized exposure during early validation.
MAX_KELLY_FRACTION = 0.025
