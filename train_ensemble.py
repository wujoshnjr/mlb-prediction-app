import pandas as pd
import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss
from datetime import datetime
import os

HISTORY_FILE = "data/historical_predictions.csv"
MODEL_OUTPUT = "data/calibrator.pkl"
TRAINING_LOG = "data/training_log.csv"
FEATURE_IMPORTANCE_LOG = "data/feature_importance.csv"

EXPECTED_FEATURES = [
    'elo_diff','market_prob','sp_era_diff','sp_fip_diff','sp_stuff_plus_diff','sp_csw_diff',
    'bullpen_ip_diff','rest_diff',
    'dynamic_park_factor',
    'platoon_ops_diff','statcast_launch_speed_diff','statcast_barrel_diff','statcast_hard_hit_diff',
    'statcast_woba_diff','timezone_diff','is_day_game','home_back2back','away_back2back',
    'catcher_era_diff','cs_diff','wind_effect',
    'temp_effect','precip_effect','injury_diff',
    'dynamic_pythag_diff','log5_prob','lag30_winrate_diff','lag30_runs_diff',
    'pitch_movement_diff',
    'k_pct_diff','bb_pct_diff','avg_bat_speed_diff',
    'pitcher_rating_diff','odds_change','odds_momentum',
    'zone_size','k_rate','bullpen_availability_diff',
    'elo_momentum_7d','elo_momentum_30d','barrel_pa_diff','hardhit_pa_diff',
    'swing_miss_diff','csw_diff','barrel_bb_pct_diff',
    'sprint_speed_diff','pitch_type_matchup_score','home_top3_woba','away_top3_woba',
    'bt_strength_diff',
    # 新增 Pitch Usage 特征
    'home_usage_magnitude','away_usage_magnitude',
    'home_shift_score','away_shift_score',
    'home_delta_FF','home_delta_SL','home_delta_CH','home_delta_CU',
    'home_delta_FC','home_delta_SI','home_delta_KC','home_delta_FS',
    'away_delta_FF','away_delta_SL','away_delta_CH','away_delta_CU',
    'away_delta_FC','away_delta_SI','away_delta_KC','away_delta_FS'
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

    # 时效性衰减权重
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
    if np.all(np.var(X, axis=0) < 1e-8):
        print("特征无方差，跳过")
        return None, None, None
    return X, y, w, df

def train():
    data = prepare_data()
    if data is None or data[0] is None: return
    X, y, w, df_all = data

    tscv = TimeSeriesSplit(n_splits=3)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    w_train = w[:split]

    # 基础学习器
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.01, random_state=42, eval_metric='logloss', use_label_encoder=False)
    lgb = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.01, random_state=42, verbose=-1)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)

    # Stacking
    estimators = [('xgb', xgb), ('lgb', lgb), ('rf', rf)]
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=tscv)

    # 双阶段校准
    calibrated_platt = CalibratedClassifierCV(estimator=stacking, method='sigmoid', cv=tscv)
    calibrated_platt.fit(X, y, sample_weight=w)

    from sklearn.isotonic import IsotonicRegression
    platt_probs = calibrated_platt.predict_proba(X)[:, 1]
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(platt_probs, y)

    class TwoStageCalibrator:
        def __init__(self, platt, iso):
            self.platt = platt; self.iso = iso
        def predict_proba(self, X):
            probs = self.platt.predict_proba(X)[:, 1]
            calibrated = self.iso.predict(probs)
            return np.column_stack([1 - calibrated, calibrated])

    final_calibrator = TwoStageCalibrator(calibrated_platt, iso_reg)
    joblib.dump(final_calibrator, MODEL_OUTPUT)
    print("Stacking 集成模型已保存")

    val_probs = final_calibrator.predict_proba(X_val)[:, 1]
    val_brier = brier_score_loss(y_val, val_probs)
    val_logloss = log_loss(y_val, val_probs)
    print(f"验证集 Brier: {val_brier:.4f}, LogLoss: {val_logloss:.4f}")

    # 记录日志（可选）
    try:
        importances = xgb.feature_importances_
        imp_df = pd.DataFrame([importances], columns=EXPECTED_FEATURES)
        imp_df['timestamp'] = datetime.now().isoformat()
        if os.path.exists(FEATURE_IMPORTANCE_LOG):
            imp_df.to_csv(FEATURE_IMPORTANCE_LOG, mode='a', header=False, index=False)
        else:
            imp_df.to_csv(FEATURE_IMPORTANCE_LOG, index=False)
    except:
        pass

if __name__ == "__main__":
    train()
