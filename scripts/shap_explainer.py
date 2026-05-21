"""
SHAP 模型可解释性模块
为单场比赛预测提供特征贡献度分析
"""
import numpy as np
import joblib

# 全局 SHAP explainer（延迟初始化）
_shap_explainer = None
_shap_model = None

def init_shap_explainer():
    """初始化 SHAP explainer（基于已训练的 XGBoost 模型）"""
    global _shap_explainer, _shap_model
    try:
        model = joblib.load("data/calibrator.pkl")
        # 从 CalibratedClassifierCV 中提取基估计器
        if hasattr(model, 'estimator_'):
            _shap_model = model.estimator_
        else:
            _shap_model = model
        import shap
        _shap_explainer = shap.TreeExplainer(_shap_model)
        print("SHAP explainer 已初始化")
        return True
    except Exception as e:
        print(f"SHAP 初始化失败: {e}")
        return False

def explain_prediction(feature_array, feature_names):
    """
    为单次预测生成 SHAP 贡献度。
    返回 {feature_name: shap_value} 字典。
    """
    global _shap_explainer
    if _shap_explainer is None:
        if not init_shap_explainer():
            return {name: 0.0 for name in feature_names}

    try:
        shap_values = _shap_explainer.shap_values(feature_array)
        # shap_values shape: (n_samples, n_features)
        contributions = {}
        for i, name in enumerate(feature_names):
            contributions[name] = round(float(shap_values[0, i]), 4)
        return contributions
    except Exception as e:
        print(f"SHAP 解释失败: {e}")
        return {name: 0.0 for name in feature_names}

def get_top_shap_features(feature_array, feature_names, top_n=5):
    """
    获取 SHAP 贡献度最大的前 N 个特征。
    返回 [(feature_name, shap_value), ...]
    """
    contributions = explain_prediction(feature_array, feature_names)
    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_contribs[:top_n]
