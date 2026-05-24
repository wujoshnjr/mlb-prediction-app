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
        print("没有预测数据")
        sys.exit(0)

    errors = []
    home_probs = [p['predicted_home_win_pct'] for p in preds]
    over_probs = [p['over_prob'] for p in preds if p.get('over_prob') is not None]
    nrfi_probs = [p['nrfi_prob'] for p in preds if p.get('nrfi_prob') is not None]
    nrfi_sources = [p.get('nrfi_source', 'unknown') for p in preds]
    pipeline_errors = data.get('errors', [])

    # 概率范围检查
    for p in home_probs:
        if not (0 <= p <= 1):
            errors.append(f"概率超出范围: {p}")
    if home_probs:
        mean_home = np.mean(home_probs)
        if mean_home > 0.58 or mean_home < 0.45:
            errors.append(f"平均主队概率异常: {mean_home:.3f}")

    if over_probs:
        mean_over = np.mean(over_probs)
        if mean_over > 0.9:
            errors.append(f"平均 over 概率过高: {mean_over:.3f}")

    # NRFI 检查：只对有效来源且非 unavailable 的检查
    valid_nrfi = [p for p in preds if p.get('nrfi_source') in ('ml', 'manual') and p.get('nrfi_prob') is not None]
    if valid_nrfi:
        prob_values = [p['nrfi_prob'] for p in valid_nrfi]
        if len(prob_values) > 2 and all(abs(x - 0.5) < 0.01 for x in prob_values):
            errors.append("NRFI 概率单一（全部≈0.5），可能模型未加载")

    if errors:
        print("❌ 验证失败:")
        for e in errors:
            print(e)
        if pipeline_errors:
            print("最近 pipeline errors:")
            for e in pipeline_errors[-5:]:
                print(f"  {e}")
        with open('report/validation_errors.txt', 'w') as f:
            f.write('\n'.join(errors))
        sys.exit(1)
    else:
        print("✅ 预测输出通过自动验证")

if __name__ == '__main__':
    validate()
