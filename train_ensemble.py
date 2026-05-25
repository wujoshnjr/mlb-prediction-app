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

try:
    import config
except ImportError:
    class config:
        MODEL_USE_MLP = False
        MODEL_META = 'lr'

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
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)

def prepare_data():
    if not os.path.exists(HISTORY_FILE):
        return None, None, None, None, None

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

    # 移除零方差特征
    var = np.var(X, axis=0)
    keep = var > 1e-8
    if not np.any(keep):
        write_status(False, True, sample_count, reason="All features zero variance")
        return None, None, None, None, None

    X = X[:, keep]
    used_features = [f for f, k in zip(EXPECTED_FEATURES, keep) if k]
    removed = [f for f, k in zip(EXPECTED_FEATURES, keep) if not k]
    if removed:
        print(f"Removed low variance features: {removed}")

    return X, y, w, df, used_features, sample_count

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

    X_test = X[calib_end:]
    y_test = y[calib_end:]
    w_test = w[calib_end:]

    # 检查 train/calib/test 是否都包含两类目标
    for name, y_set in [("train", y_train), ("calib", y_calib), ("test", y_test)]:
        unique = np.unique(y_set)
        if len(unique) < 2:
            write_status(False, True, sample_count, reason=f"{name} set contains only one class, skipped")
            return

    print(f"训练集: {len(X_train)}  校准集: {len(X_calib)}  测试集: {len(X_test)}")

    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.01,
                         importance_type='gain', random_state=42,
                         eval_metric='logloss', use_label_encoder=False)
    lgb = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.01,
                          random_state=42, verbose=-1)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
    estimators = [('xgb', xgb), ('lgb', lgb), ('rf', rf)]

    if config.MODEL_USE_MLP:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                            alpha=0.001, early_stopping=True, random_state=42)
        mlp_pipe = Pipeline([('scaler', StandardScaler()), ('mlp', mlp)])
        estimators.append(('mlp', mlp_pipe))

    if config.MODEL_META == 'elasticnet':
        from sklearn.linear_model import SGDClassifier
        final_estimator = SGDClassifier(loss='log_loss', penalty='elasticnet',
                                        l1_ratio=0.5, alpha=0.0001,
                                        random_state=42, max_iter=2000, tol=1e-3)
    else:
        final_estimator = LogisticRegression(random_state=42, max_iter=2000)

    stacking = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5)
    stacking.fit(X_train, y_train, sample_weight=w_train)

    calibrated = CalibratedClassifierCV(estimator=stacking, method='sigmoid', cv='prefit')
    calibrated.fit(X_calib, y_calib, sample_weight=w_calib)

    # 测试集评估
    test_probs = calibrated.predict_proba(X_test)[:, 1]
    test_brier = brier_score_loss(y_test, test_probs)
    test_logloss = log_loss(y_test, test_probs)

    # 保存 artifact
    artifact = {
        "model": calibrated,
        "features": used_features,
        "schema_version": "v1",
        "trained_at": datetime.now().isoformat(),
        "training_sample_count": sample_count
    }
    joblib.dump(artifact, MODEL_OUTPUT)

    write_status(True, False, sample_count,
                 brier=round(test_brier, 4),
                 logloss=round(test_logloss, 4))

    # 日志
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(df_all),
        "brier": round(test_brier, 4),
        "logloss": round(test_logloss, 4)
    }
    pd.DataFrame([log_entry]).to_csv(TRAINING_LOG,
                                     mode='a',
                                     header=not os.path.exists(TRAINING_LOG),
                                     index=False)

    # 特征重要性
    xgb_imp = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.01,
                            importance_type='gain', random_state=42)
    xgb_imp.fit(X_train, y_train, sample_weight=w_train)
    importances = xgb_imp.feature_importances_
    feat_names = used_features[:len(importances)]
    imp_df = pd.DataFrame([importances], columns=feat_names)
    imp_df['timestamp'] = datetime.now().isoformat()
    imp_df.to_csv(FEATURE_IMPORTANCE_LOG,
                  mode='a',
                  header=not os.path.exists(FEATURE_IMPORTANCE_LOG),
                  index=False)

    sorted_idx = np.argsort(importances)
    print("\n重要性最低的5个特征:")
    for i in sorted_idx[:5]:
        print(f"  {feat_names[i]}: {importances[i]:.6f}")

if __name__ == "__main__":
    train()
