import pandas as pd
import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss
from datetime import datetime

HISTORY_FILE = "data/historical_predictions.csv"
MODEL_OUTPUT = "data/calibrator.pkl"
TRAINING_LOG = "data/training_log.csv"

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
    'elo_momentum_7d','elo_momentum_30d','barrel_pa_diff','hardhit_pa_diff',
    'swing_miss_diff',        # 新增
    'csw_diff',               # 新增
    'barrel_bb_pct_diff'      # 新增
]

def prepare_data():
    df = pd.read_csv(HISTORY_FILE)
    if 'game_date' in df.columns:
        df = df.sort_values('game_date').reset_index(drop=True)

    df['home_win'] = df['home_win'].replace('', np.nan)
    df = df.dropna(subset=['home_win'])
    df['home_win'] = df['home_win'].astype(int)

    if len(df) < 50:
        print(f"数据量不足 ({len(df)})，跳过训练")
        return None, None, None

    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    X = df[EXPECTED_FEATURES].values
    y = df['home_win'].values
    if np.all(np.var(X, axis=0) < 1e-8):
        print("特征无方差，跳过")
        return None, None, None
    return X, y, df

def train():
    X, y, df_all = prepare_data()
    if X is None: return

    tscv = TimeSeriesSplit(n_splits=3)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # XGBoost 超参数优化
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

    # LightGBM 超参数优化
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

    # 软投票集成
    ensemble = VotingClassifier(estimators=[('xgb', best_xgb), ('lgb', best_lgb)], voting='soft')

    # 双阶段校准 (Platt + Isotonic)
    calibrated_platt = CalibratedClassifierCV(estimator=ensemble, method='sigmoid', cv=tscv)
    calibrated_platt.fit(X, y)

    from sklearn.isotonic import IsotonicRegression
    platt_probs = calibrated_platt.predict_proba(X)[:, 1]
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(platt_probs, y)

    class TwoStageCalibrator:
        def __init__(self, platt, iso):
            self.platt = platt
            self.iso = iso
        def predict_proba(self, X):
            probs = self.platt.predict_proba(X)[:, 1]
            calibrated_probs = self.iso.predict(probs)
            return np.column_stack([1 - calibrated_probs, calibrated_probs])

    final_calibrator = TwoStageCalibrator(calibrated_platt, iso_reg)
    joblib.dump(final_calibrator, MODEL_OUTPUT)
    print(f"双阶段校准模型已保存至 {MODEL_OUTPUT}")

    # 评估
    val_probs = final_calibrator.predict_proba(X_val)[:, 1]
    val_brier = brier_score_loss(y_val, val_probs)
    val_logloss = log_loss(y_val, val_probs)
    print(f"验证集 Brier: {val_brier:.4f}, LogLoss: {val_logloss:.4f}")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(df_all),
        "brier": round(val_brier, 4),
        "logloss": round(val_logloss, 4),
        "xgb_params": study_xgb.best_params,
        "lgb_params": study_lgb.best_params
    }
    log_df = pd.DataFrame([log_entry])
    if os.path.exists(TRAINING_LOG):
        log_df.to_csv(TRAINING_LOG, mode='a', header=False, index=False)
    else:
        log_df.to_csv(TRAINING_LOG, index=False)
    print(f"训练日志已追加至 {TRAINING_LOG}")

    try:
        importances = best_xgb.feature_importances_
        for name, imp in zip(EXPECTED_FEATURES, importances):
            print(f"{name}: {imp:.4f}")
    except Exception as e:
        print(f"特征重要性错误: {e}")

if __name__ == "__main__":
    import os
    train()
