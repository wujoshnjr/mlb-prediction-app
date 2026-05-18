import pandas as pd
import numpy as np
from datetime import datetime

def run_backtest(predictions_file="report/prediction.json", historical_results_file=None):
    """
    简易回测：假设 historical_results_file 包含 'game_id','home_score','away_score'
    若没有，则跳过实际结果比对，仅统计推荐次数。
    """
    import json, os
    if not os.path.exists(predictions_file):
        print("预测文件不存在")
        return

    with open(predictions_file, 'r') as f:
        data = json.load(f)

    preds = data.get("today_predictions", [])
    # 统计推荐数量
    ml = [p for p in preds if p.get("moneyline_recommendation") != "PASS"]
    spread = [p for p in preds if p.get("spread_recommendation") != "PASS"]
    total = [p for p in preds if p.get("total_recommendation") != "PASS"]

    report = {
        "date": datetime.now().isoformat(),
        "total_games": len(preds),
        "moneyline_recommendations": len(ml),
        "spread_recommendations": len(spread),
        "total_recommendations": len(total),
    }
    # 如果有实际结果文件，计算ROI、CLV等
    if historical_results_file and os.path.exists(historical_results_file):
        results_df = pd.read_csv(historical_results_file)  # 假设有 game_id, home_score, away_score, closing_odds...
        # 实现与预测数据的合并和ROI计算
        # 此处省略详细实现，后续可扩展
        pass

    print("回测报告：", json.dumps(report, indent=2))
    return report
