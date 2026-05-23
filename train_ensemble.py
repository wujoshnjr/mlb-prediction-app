# train_ensemble.py
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss

# 导入配置（防御性）
try:
    import config
except ImportError:
    class config:
        MODEL_USE_MLP = False
        MODEL_META = 'lr'

# 可选模块
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except ImportError:
    MLPClassifier = None
    StandardScaler = None
    Pipeline = None

try:
    from sklearn.linear_model import SGDClassifier
except ImportError:
    SGDClassifier = None

HISTORY_FILE = "data/historical_predictions.csv"
MODEL_OUTPUT = "data/calibrator.pkl"
TRAINING_LOG = "data/training_log.csv"
FEATURE_IMPORTANCE_LOG = "data/feature_importance.csv"

# 与最新 prediction.py 完全同步的特征列表（无 market_prob，全部 diff）
EXPECTED_FEATURES = [
    'elo_diff',
    'sp_era_diff', 'sp_fip_diff', 'sp_stuff_plus_diff', 'sp_csw_diff',
    'bullpen_ip_diff', 'rest_diff',
    'dynamic_park_factor',
    'platoon_ops_diff', 'statcast_launch_speed_diff', 'statcast_barrel_diff',
    'statcast_hard_hit_diff', 'statcast_woba_diff',
    'timezone_diff', 'is_day_game', 'back2back_diff',   # 替换 home_back2back/away_back2back
    'catcher_era_diff', 'cs_diff', 'wind_effect',
    'temp_effect', 'precip_effect', 'injury_diff',
    'dynamic_pythag_diff', 'log5_prob', 'lag30_winrate_diff', 'lag30_runs_diff',
    'pitch_movement_diff',
    'k_pct_diff', 'bb_pct_diff', 'avg_bat_speed_diff',
    'pitcher_rating_diff', 'odds_change', 'odds_momentum',
    'zone_size', 'k_rate', 'bullpen_availability_diff',
    'elo_momentum_7d', 'elo_momentum_30d', 'barrel_pa_diff', 'hardhit_pa_diff',
    'swing_miss_diff', 'csw_diff', 'barrel_bb_pct_diff',
    'sprint_speed_diff', 'pitch_type_matchup_score',
    'top3_woba_diff',   # 新：前三棒 wOBA 差值
    'winrate_diff',     # 新：胜率差值
    'bt_strength_diff',
    # Pitch Usage 特征
    'home_usage_magnitude', 'away_usage_magnitude',
    'home_shift_score', 'away_shift_score',
    'home_delta_FF', 'home_delta_SL', 'home_delta_CH', 'home_delta_CU',
    'home_delta_FC', 'home_delta_SI', 'home_delta_KC', 'home_delta_FS',
    'away_delta_FF', 'away_delta_SL', 'away_delta_CH', 'away_delta_CU',
    'away_delta_FC', 'away_delta_SI', 'away_delta_KC', 'away_delta_FS'
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
        return None, None, None, None, None

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

    # 移除低方差特征
    var = np.var(X, axis=0)
    keep_mask = var > 1e-8
    if not np.any(keep_mask):
        print("所有特征均无方差，跳过训练")
        return None, None, None, None, None

    X = X[:, keep_mask]
    kept_features = [f for f, k in zip(EXPECTED_FEATURES, keep_mask) if k]
    removed_features = [f for f, k in zip(EXPECTED_FEATURES, keep_mask) if not k]
    if removed_features:
        print(f"已移除无方差特征: {removed_features}")

    return X, y, w, df, kept_features

def train():
    data = prepare_data()
    if data is None or data[0] is None:
        return
    X, y, w, df_all, used_features = data

    # 严格时间切分：70% 训练 / 15% 校准 / 15% 测试
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

    print(f"训练集: {len(X_train)}  校准集: {len(X_calib)}  测试集: {len(X_test)}")

    # 基础学习器
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.01,
                         importance_type='gain', random_state=42,
                         eval_metric='logloss', use_label_encoder=False)
    lgb = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.01,
                          random_state=42, verbose=-1)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
    estimators = [('xgb', xgb), ('lgb', lgb), ('rf', rf)]

    # 可选 MLP
    if config.MODEL_USE_MLP:
        if MLPClassifier is not None and StandardScaler is not None and Pipeline is not None:
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                                alpha=0.001, early_stopping=True, random_state=42)
            mlp_pipe = Pipeline([('scaler', StandardScaler()), ('mlp', mlp)])
            estimators.append(('mlp', mlp_pipe))

    # 元学习器
    if config.MODEL_META == 'elasticnet' and SGDClassifier is not None:
        final_estimator = SGDClassifier(loss='log_loss', penalty='elasticnet',
                                        l1_ratio=0.5, alpha=0.0001,
                                        random_state=42, max_iter=2000, tol=1e-3)
    else:
        final_estimator = LogisticRegression(random_state=42, max_iter=2000)

    stacking = StackingClassifier(estimators=estimators,
                                  final_estimator=final_estimator,
                                  cv='prefit')

    # 训练 Stacking
    print("训练 Stacking 模型...")
    stacking.fit(X_train, y_train, sample_weight=w_train)

    # 在独立校准集上做 sigmoid 校准
    print("进行 sigmoid 校准...")
    calibrated_model = CalibratedClassifierCV(estimator=stacking, method='sigmoid',
                                              cv='prefit')
    calibrated_model.fit(X_calib, y_calib, sample_weight=w_calib)

    # 保存最终模型
    joblib.dump(calibrated_model, MODEL_OUTPUT)
    print("模型已保存至", MODEL_OUTPUT)

    # 测试集评估
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]
    test_brier = brier_score_loss(y_test, test_probs)
    test_logloss = log_loss(y_test, test_probs)
    print(f"测试集 Brier: {test_brier:.4f}, LogLoss: {test_logloss:.4f}")

    # 训练日志
    log_entry = {"timestamp": datetime.now().isoformat(),
                 "num_samples": len(df_all),
                 "brier": round(test_brier, 4),
                 "logloss": round(test_logloss, 4)}
    log_df = pd.DataFrame([log_entry])
    if os.path.exists(TRAINING_LOG):
        log_df.to_csv(TRAINING_LOG, mode='a', header=False, index=False)
    else:
        log_df.to_csv(TRAINING_LOG, index=False)

    # 特征重要性（基于 XGBoost gain）
    xgb_for_imp = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.01,
                                importance_type='gain', random_state=42,
                                eval_metric='logloss', use_label_encoder=False)
    xgb_for_imp.fit(X_train, y_train, sample_weight=w_train)
    importances = xgb_for_imp.feature_importances_

    feat_names = used_features[:len(importances)] if len(importances) <= len(used_features) else used_features
    imp_df = pd.DataFrame([importances], columns=feat_names)
    imp_df['timestamp'] = datetime.now().isoformat()
    if os.path.exists(FEATURE_IMPORTANCE_LOG):
        imp_df.to_csv(FEATURE_IMPORTANCE_LOG, mode='a', header=False, index=False)
    else:
        imp_df.to_csv(FEATURE_IMPORTANCE_LOG, index=False)

    sorted_idx = np.argsort(importances)
    print("\n⚠️ 重要性最低的5个特征:")
    for i in sorted_idx[:5]:
        if i < len(feat_names):
            print(f"  {feat_names[i]}: {importances[i]:.6f}")

if __name__ == "__main__":
    train()
