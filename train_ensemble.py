# train_ensemble.py (新版训练流程)
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import config

def train_ensemble_model(X, y, sample_weights=None, n_splits=5):
    """
    使用时间序列交叉验证训练Stacking集成模型，支持ElasticNet元学习器和MLP。
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = []

    # 基础学习器
    base_estimators = [
        ('xgb', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                  use_label_encoder=False, random_state=42)),
        ('lgb', lgb.LGBMClassifier(objective='binary', random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ]

    # 可选MLP
    if config.MODEL_USE_MLP:
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                            alpha=0.001, early_stopping=True, random_state=42)
        # 需要标准化，通过管道
        from sklearn.pipeline import Pipeline
        mlp_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', mlp)
        ])
        base_estimators.append(('mlp', mlp_pipe))

    # 元学习器
    if config.MODEL_META == 'elasticnet':
        # 注意ElasticNet是回归器，但我们需要概率输出，所以用ElasticNetCV + sigmoid转换
        # 或者直接使用逻辑回归作为元学习器，ElasticNet惩罚可通过SGDClassifier实现
        from sklearn.linear_model import SGDClassifier
        meta = SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.5,
                             alpha=0.0001, random_state=42, max_iter=1000)
    else:
        meta = LogisticRegression(random_state=42, max_iter=1000)

    # 整体Stacking
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        cv=tscv,                    # 内层也使用时序交叉验证
        passthrough=False,
        stack_method='predict_proba'
    )

    # 训练整个模型
    stacking.fit(X, y, sample_weight=sample_weights)
    return stacking

# 训练脚本入口（示例）
if __name__ == '__main__':
    # 加载数据...
    # data = pd.read_csv('data/training_features.csv')
    # X = data.drop(['target'], axis=1)
    # y = data['target']
    # 计算时效性样本权重
    # dates = pd.to_datetime(data['date'])
    # recent_weight = np.exp(-(dates.max() - dates).dt.days / 365)  # 简单指数衰减
    # model = train_ensemble_model(X, y, sample_weights=recent_weight)
    # joblib.dump(model, 'models/stacking_model.pkl')
    pass
