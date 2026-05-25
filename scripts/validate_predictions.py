# scripts/validate_predictions.py
import sys, os, json
import numpy as np

REPORT = 'report/prediction.json'

def validate():
    if not os.path.exists(REPORT):
        print(f"❌ {REPORT} 不存在")
        sys.exit(1)

    with open(REPORT) as f:
        data = json.load(f)

    preds = data.get('today_predictions', [])
    if not preds:
        print("没有预测数据，检查是否有比赛")
        # 可进一步检查 schedule，但暂时 pass
        sys.exit(0)

    home_probs = [p['predicted_home_win_pct'] for p in preds]
    mean_home = np.mean(home_probs)
    min_home = np.min(home_probs)
    max_home = np.max(home_probs)

    errors = []
    if len(preds) >= 5:
        if mean_home > 0.60 or mean_home < 0.40:
            errors.append(f"Avg home prob {mean_home:.3f} 异常")
    else:
        print(f"仅 {len(preds)} 场比赛，跳过平均检查")

    # Rating 范围检查
    elo_file = 'data/elo_ratings.json'
    if os.path.exists(elo_file):
        with open(elo_file) as f:
            elo = json.load(f)
        if elo:
            ratings = list(elo.values())
            rating_range = max(ratings) - min(ratings)
            if rating_range > 400:
                errors.append(f"Elo range {rating_range:.0f} 过大")

    # 模型来源分布
    sources = [p.get('model_source','unknown') for p in preds]
    if all(s == 'manual' for s in sources):
        print("所有比赛使用手工预测")

    if errors:
        print("❌ 验证失败:")
        for e in errors: print(e)
        with open('report/validation_errors.txt','w') as f: f.write('\n'.join(errors))
        sys.exit(1)
    else:
        print("✅ 预测通过验证")

if __name__ == '__main__':
    validate()
