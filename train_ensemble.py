import pandas as pd
import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

HISTORY_FILE = "data/historical_predictions.csv"
MODEL_OUTPUT = "data/calibrator.pkl"

EXPECTED_FEATURES = [
    'elo_diff','market_prob','sp_era_diff','sp_fip_diff','bullpen_ip_diff','rest_diff',
    'dynamic_park_factor',
    'platoon_ops_diff','statcast_launch_speed_diff','statcast_barrel_diff','statcast_hard_hit_diff',
    'statcast_woba_diff','timezone_diff','is_day_game','home_back2back','away_back2back',
    'catcher_era_diff','cs_diff','wind_effect',
    'temp_effect','precip_effect','injury_diff',
    'pythag_diff','log5_prob','lag30_winrate_diff','lag30_runs_diff',
    'pitch_movement_diff',
    'k_pct_diff','bb_pct_diff','avg_bat_speed_diff',
    'pitcher_rating_diff','odds_change',
    'zone_size','k_rate','bullpen_availability_diff',
    'elo_momentum_7d','elo_momentum_30d','barrel_pa_diff','hardhit_pa_diff'
]

def prepare_data():
    df = pd.read_csv(HISTORY_FILE)
    df['home_win'] = df['home_win'].replace('', np.nan)
    df = df.dropna(subset=['home_win'])
    df['home_win'] = df['home_win'].astype(int)
    if len(df) < 50:
        print(f"数据量不足 ({len(df)})，跳过训练")
        return None, None
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    X = df[EXPECTED_FEATURES].values
    y = df['home_win'].values
    if np.all(np.var(X, axis=0) < 1e-8):
        print("特征无方差，跳过")
        return None, None
    return X, y

def train():
    X, y = prepare_data()
    if X is None: return

    tscv = TimeSeriesSplit(n_splits=3)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:,1]
        return log_loss(y_val, preds)

    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=False)
    best_xgb = XGBClassifier(**study_xgb.best_params, eval_metric='logloss', random_state=42)
    print("XGBoost 最佳参数:", study_xgb.best_params)

    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'random_state': 42, 'verbose': -1
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:,1]
        return log_loss(y_val, preds)

    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=False)
    best_lgb = LGBMClassifier(**study_lgb.best_params, verbose=-1, random_state=42)
    print("LightGBM 最佳参数:", study_lgb.best_params)

    ensemble = VotingClassifier(estimators=[('xgb', best_xgb), ('lgb', best_lgb)], voting='soft')
    calibrated = CalibratedClassifierCV(estimator=ensemble, method='isotonic', cv=tscv)
    calibrated.fit(X, y)

    joblib.dump(calibrated, MODEL_OUTPUT)
    print(f"集成模型已保存至 {MODEL_OUTPUT}")

    try:
        importances = best_xgb.feature_importances_
        for name, imp in zip(EXPECTED_FEATURES, importances):
            print(f"{name}: {imp:.4f}")
    except Exception as e:
        print(f"特征重要性错误: {e}")

if __name__ == "__main__":
    train()
