# scripts/shap_explainer.py
"""
SHAP 可解释性模块 — 对 Stacking 集成模型进行近似解释。
支持树模型 (TreeExplainer) 和神经网络 (KernelExplainer) 的混合解释，
通过平均第一层各模型 SHAP 值来获得整体特征重要性。

依赖：pip install shap matplotlib
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def explain_stacking(model, X_background, X_sample, feature_names, class_names=None,
                     output_path='shap_summary.png'):
    """
    对 Stacking 集成模型计算近似 SHAP 值并生成汇总图。
    
    参数：
        model: 已训练的 StackingClassifier (sklearn)
        X_background: DataFrame/array，背景数据集（通常取训练集抽样 100 行）
        X_sample: DataFrame/array，要解释的样本集
        feature_names: list，特征名称
        class_names: list，类别名称，默认 ['Lose', 'Win']
        output_path: str，保存路径
    
    返回：
        avg_shap_values: np.array，形状与 X_sample 相同，平均 SHAP 值
    """
    if class_names is None:
        class_names = ['Lose', 'Win']
    
    # 确保输入为 DataFrame 以保留列名
    if not isinstance(X_background, pd.DataFrame):
        X_background = pd.DataFrame(X_background, columns=feature_names)
    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)
    
    # 获取第一层估算器
    estimators = model.named_estimators_
    shap_values_list = []
    
    # 确定目标类别索引（二分类正类为 1）
    target_class_idx = 1  # 假设 predict_proba 的第二列是正类
    
    for name, est in estimators.items():
        print(f"正在解释第一层学习器: {name}")
        try:
            # 判断模型类型并选择合适的解释器
            if hasattr(est, 'get_booster'):          # XGBoost
                explainer = shap.TreeExplainer(est)
                shap_val = explainer.shap_values(X_sample)
            elif hasattr(est, 'booster_'):           # LightGBM
                explainer = shap.TreeExplainer(est)
                shap_val = explainer.shap_values(X_sample)
            elif hasattr(est, 'estimators_'):        # Random Forest
                explainer = shap.TreeExplainer(est)
                shap_val = explainer.shap_values(X_sample)
            else:
                # 使用 KernelExplainer 近似（适合 MLP 等非树模型）
                # 注意：KernelExplainer 较慢，背景数据量要小
                print(f"  使用 KernelExplainer (背景数据 {X_background.shape[0]} 行)...")
                explainer = shap.KernelExplainer(est.predict_proba, X_background)
                shap_val = explainer.shap_values(X_sample, nsamples=100)
            
            # 处理返回格式：某些模型返回 list of arrays (每个类一个)
            if isinstance(shap_val, list) and len(shap_val) == 2:
                shap_positive = shap_val[target_class_idx]
            else:
                shap_positive = shap_val
            
            shap_values_list.append(shap_positive)
            print(f"  完成，形状: {shap_positive.shape}")
        
        except Exception as e:
            print(f"  跳过 {name}，原因: {str(e)}")
            continue
    
    if not shap_values_list:
        print("没有成功计算任何 SHAP 值，返回全零数组。")
        return np.zeros(X_sample.shape)
    
    # 计算平均 SHAP 值（按元素平均）
    avg_shap_values = np.mean(np.stack(shap_values_list, axis=0), axis=0)
    
    # 生成汇总图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(avg_shap_values, X_sample, feature_names=feature_names,
                      show=False, class_names=class_names)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP 汇总图已保存至: {output_path}")
    
    # 输出特征重要性排序（基于平均绝对值）
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(avg_shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    print("\n特征重要性 (Top 15):")
    print(importance_df.head(15).to_string(index=False))
    
    return avg_shap_values


# ========== 独立测试入口 ==========
if __name__ == '__main__':
    print("SHAP 解释器模块加载成功。")
    print("示例用法（在训练后调用）：")
    print("""
    import joblib
    from scripts.shap_explainer import explain_stacking
    
    # 加载模型和特征名
    model = joblib.load('models/stacking_model.pkl')
    X_train = pd.read_csv('data/training_features.csv')
    feature_names = X_train.columns.tolist()
    
    # 背景数据和待解释样本
    X_bg = X_train.sample(min(100, len(X_train)), random_state=42)
    X_sample = X_train.sample(50, random_state=43)
    
    # 计算并绘图
    shap_vals = explain_stacking(model, X_bg, X_sample, feature_names)
    """)
