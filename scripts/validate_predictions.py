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
    errors = data.get('errors', [])

    # 检查是否应该无比赛
    if not preds:
        schedule_errors = [e for e in errors if 'schedule' in e.lower() or 'no games' in e.lower()]
        if schedule_errors:
            print("今日无比赛或赛程错误，跳过验证")
            sys.exit(0)
        else:
            print("❌ 预测为空但没有 schedule 错误信息，可能数据获取失败")
            sys.exit(1)

    home_probs = [p['predicted_home_win_pct'] for p in preds]
    mean_home = np.mean(home_probs)
    min_home = np.min(home_probs)
    max_home = np.max(home_probs)

    validation_errors = []
    if len(preds) >= 5:
        if mean_home > 0.60 or mean_home < 0.40:
            validation_errors.append(f"Avg home prob {mean_home:.3f} 异常")
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
                validation_errors.append(f"Elo range {rating_range:.0f} 过大")

    # 模型来源分布
    sources = [p.get('model_source', 'unknown') for p in preds]
    if all(s == 'manual' for s in sources) and len(preds) > 0:
        print("所有比赛使用手工预测")

    # NRFI 检查
    nrfi_probs = [p['nrfi_prob'] for p in preds if p.get('nrfi_prob') is not None]
    nrfi_sources = [p.get('nrfi_source', 'unknown') for p in preds]
    if nrfi_probs:
        valid_nrfi = [p for p in preds if p.get('nrfi_source') in ('ml', 'manual') and p.get('nrfi_prob') is not None]
        if valid_nrfi and len(valid_nrfi) > 2 and all(abs(p['nrfi_prob'] - 0.5) < 0.01 for p in valid_nrfi):
            validation_errors.append("NRFI 概率单一（全部≈0.5）")

    if validation_errors:
        print("❌ 验证失败:")
        for e in validation_errors:
            print(e)
        if errors:
            print("pipeline errors 摘要:")
            for e in errors[-5:]:
                print(f"  {e}")
        with open('report/validation_errors.txt', 'w') as f:
            f.write('\n'.join(validation_errors))
        sys.exit(1)
    else:
        print("✅ 预测输出通过自动验证")

if __name__ == '__main__':
    validate()
