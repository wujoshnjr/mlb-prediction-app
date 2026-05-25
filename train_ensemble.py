# train_ensemble.py
import pandas as pd
import numpy as np
import joblib, os, json
from datetime import datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
from scripts.feature_schema import EXPECTED_FEATURES

try: import config
except: class config: MODEL_USE_MLP=False; MODEL_META='lr'

HISTORY_FILE = "data/historical_predictions.csv"
MODEL_OUTPUT = "data/calibrator.pkl"
STATUS_FILE = "data/training_status.json"
TRAINING_LOG = "data/training_log.csv"
FEATURE_IMPORTANCE_LOG = "data/feature_importance.csv"

MIN_TRAIN_SAMPLES = 100

def write_status(trained, skipped, sample_count, reason=None, brier=None, logloss=None):
    status = {
        "trained": trained,
        "skipped": skipped,
        "sample_count": sample_count,
        "reason": reason,
        "brier": brier,
        "logloss": logloss,
        "timestamp": datetime.now().isoformat()
    }
    with open(STATUS_FILE,'w') as f: json.dump(status, f, indent=2)

def prepare_data():
    df = pd.read_csv(HISTORY_FILE)
    if 'game_date' in df.columns:
        df = df.sort_values('game_date').reset_index(drop=True)
    df['home_win'] = df['home_win'].replace('', np.nan)
    df = df.dropna(subset=['home_win'])
    df['home_win'] = df['home_win'].astype(int)
    sample_count = len(df)
    if sample_count < MIN_TRAIN_SAMPLES:
        write_status(False, True, sample_count, reason=f"Insufficient samples ({sample_count} < {MIN_TRAIN_SAMPLES})")
        return None, None, None, None, None

    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    if 'game_date' in df.columns:
        max_date = pd.to_datetime(df['game_date']).max()
        df['days_ago'] = (max_date - pd.to_datetime(df['game_date'])).dt.days
        df['sample_weight'] = np.exp(-df['days_ago'] / 365 * np.log(2))
        df['sample_weight'] = df['sample_weight'].clip(lower=0.1)
    else:
        df['sample_weight'] = 1.0

    X = df[EXPECTED_FEATURES].values
    y = df['home_win'].values
    w = df['sample_weight'].values

    var = np.var(X, axis=0)
    keep = var > 1e-8
    if not np.any(keep):
        write_status(False, True, sample_count, reason="All features zero variance")
        return None, None, None, None, None

    X = X[:, keep]
    kept_features = [f for f, k in zip(EXPECTED_FEATURES, keep) if k]
    removed = [f for f, k in zip(EXPECTED_FEATURES, keep) if not k]
    if removed: print(f"Removed low variance features: {removed}")

    return X, y, w, df, kept_features, sample_count

def train():
    data = prepare_data()
    if data is None or data[0] is None:
        return
    X, y, w, df_all, used_features, sample_count = data

    n = len(X)
    train_end = int(n * 0.7)
    calib_end = int(n * 0.85)

    X_train = X[:train_end]
    y_train = y[:train_end]
    w_train = w[:train_end]

    X_calib = X[train_end:calib_end]
    y_calib = y[train_end:calib_end]
    w_calib = w[train_end:calib_end]

    print(f"训练集: {len(X_train)}  校准集: {len(X_calib)}")

    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.01, importance_type='gain', random_state=42, eval_metric='logloss')
    lgb = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.01, random_state=42, verbose=-1)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
    estimators = [('xgb', xgb), ('lgb', lgb), ('rf', rf)]

    final_estimator = LogisticRegression(random_state=42, max_iter=2000)

    stacking = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5)
    stacking.fit(X_train, y_train, sample_weight=w_train)

    calibrated = CalibratedClassifierCV(estimator=stacking, method='sigmoid', cv='prefit')
    calibrated.fit(X_calib, y_calib, sample_weight=w_calib)

    # Smoke test
    try:
        _ = calibrated.predict_proba(X_calib[:1])
        print("Smoke test passed")
    except Exception as e:
        write_status(False, True, sample_count, reason=f"Smoke test failed: {e}")
        return

    artifact = {
        "model": calibrated,
        "features": used_features,
        "schema_version": "v1",
        "trained_at": datetime.now().isoformat(),
        "training_sample_count": sample_count
    }
    joblib.dump(artifact, MODEL_OUTPUT)

    # 评估
    test_probs = calibrated.predict_proba(X_calib)[:, 1]
    test_brier = brier_score_loss(y_calib, test_probs)
    test_logloss = log_loss(y_calib, test_probs)
    write_status(True, False, sample_count, brier=round(test_brier,4), logloss=round(test_logloss,4))
    print(f"Model saved, features: {len(used_features)}")

    # 日志
    log_entry = {"timestamp": datetime.now().isoformat(), "num_samples": len(df_all), "brier": round(test_brier,4), "logloss": round(test_logloss,4)}
    pd.DataFrame([log_entry]).to_csv(TRAINING_LOG, mode='a', header=not os.path.exists(TRAINING_LOG), index=False)

    # 特征重要性
    xgb_imp = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.01, importance_type='gain', random_state=42)
    xgb_imp.fit(X_train, y_train, sample_weight=w_train)
    importances = xgb_imp.feature_importances_
    feat_names = used_features[:len(importances)]
    imp_df = pd.DataFrame([importances], columns=feat_names)
    imp_df['timestamp'] = datetime.now().isoformat()
    imp_df.to_csv(FEATURE_IMPORTANCE_LOG, mode='a', header=not os.path.exists(FEATURE_IMPORTANCE_LOG), index=False)

if __name__ == "__main__":
    train()
