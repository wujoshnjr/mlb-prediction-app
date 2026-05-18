"""
训练概率校准器（Isotonic Regression）

使用方法：
  1. 收集一段时间的预测结果和真实比赛结果。
  2. 准备两个数组：raw_probs（模型预测的主队获胜概率）和 actual（实际主队是否获胜，0/1）。
  3. 运行此脚本，生成 data/calibrator.pkl。
"""
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
import pandas as pd


def load_data(predictions_file, results_file):
    """
    从 predictions.json 和实际结果 CSV 中加载数据。
    predictions.json：由 generate_predictions 生成的每日预测
    results_file：包含 game_id, home_win (0/1) 的 CSV
    """
    import json
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    preds = data.get("today_predictions", [])
    
    raw_probs = []
    actuals = []
    
    # 读取结果
    results_df = pd.read_csv(results_file)  # 必须包含 game_id 和 home_win
    for p in preds:
        game_id = p.get("game_id")  # 需要确保 prediction 中包含 game_id 字段
        if game_id is None:
            continue
        result_row = results_df[results_df['game_id'] == game_id]
        if not result_row.empty:
            raw_probs.append(p['predicted_home_win_pct'])
            actuals.append(result_row.iloc[0]['home_win'])
    
    return np.array(raw_probs), np.array(actuals)


def train_calibrator(raw_probs, actuals, save_path="data/calibrator.pkl"):
    """训练并保存校准器"""
    if len(raw_probs) < 50:
        print("数据量不足（至少需要50条），无法训练校准器")
        return None

    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probs, actuals)
    joblib.dump(calibrator, save_path)
    print(f"校准器已保存至 {save_path}")
    return calibrator


if __name__ == "__main__":
    # 示例用法（你需要根据实际情况修改文件路径）
    # 你可以指定一个预测文件和一个结果文件
    pred_file = "report/prediction.json"  # 某个历史预测文件
    result_file = "data/historical_results.csv"  # 你收集的实际结果文件

    # 如果没有实际结果文件，可以手动提供数组进行演示
    # 这里给出一个模拟训练示例，实际使用时请注释掉并替换为真实数据加载
    print("注意：本示例使用模拟数据，实际训练请替换为真实数据。")
    # 模拟数据
    np.random.seed(42)
    raw_probs = np.random.beta(6, 4, 200)  # 模拟模型预测概率
    actuals = (raw_probs + np.random.normal(0, 0.1, 200)) > 0.5
    actuals = actuals.astype(int)

    train_calibrator(raw_probs, actuals)
