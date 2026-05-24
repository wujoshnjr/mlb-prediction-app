# scripts/validate_predictions.py
import sys, os, json
import numpy as np

REPORT = 'report/prediction.json'
if not os.path.exists(REPORT):
    sys.exit(0)

with open(REPORT) as f:
    data = json.load(f)

preds = data.get('today_predictions', [])
if not preds:
    sys.exit(0)

home_probs = [p['predicted_home_win_pct'] for p in preds]
over_probs = [p['over_prob'] for p in preds if p.get('over_prob') is not None]
nrfi_probs = [p['nrfi_prob'] for p in preds if p.get('nrfi_prob') is not None]
nrfi_sources = [p.get('nrfi_source', 'unknown') for p in preds]
errors = data.get('errors', [])

diagnostics = {
    'avg_home_prob': np.mean(home_probs) if home_probs else None,
    'min_home_prob': np.min(home_probs) if home_probs else None,
    'max_home_prob': np.max(home_probs) if home_probs else None,
    'extreme_home': [p['game_id'] for p in preds if p['predicted_home_win_pct'] > 0.80 or p['predicted_home_win_pct'] < 0.20],
    'nrfi_source_distribution': dict(zip(*np.unique(nrfi_sources, return_counts=True))) if nrfi_sources else {},
    'pipeline_errors': errors[-5:] if errors else [],
}

validation_errors = []
if home_probs:
    mean_home = diagnostics['avg_home_prob']
    if mean_home > 0.58 or mean_home < 0.45:
        validation_errors.append(f"Avg home prob = {mean_home:.3f}, 偏离正常区间")
        validation_errors.append(f"  分布: min={diagnostics['min_home_prob']:.3f} max={diagnostics['max_home_prob']:.3f}")
        if diagnostics['extreme_home']:
            validation_errors.append(f"  极端比赛: {diagnostics['extreme_home'][:5]}")

if over_probs:
    mean_over = np.mean(over_probs)
    if mean_over > 0.9:
        validation_errors.append(f"Mean over prob = {mean_over:.3f}, 仿真可能崩溃")

# NRFI 检查：只有当有有效 ML 预测且全部为 0.5 时才报错
if nrfi_probs:
    # 如果所有 nrfi_source 都是 'unavailable'，则跳过
    if all(s == 'unavailable' for s in nrfi_sources):
        pass  # 数据不可用，不判失败
    elif all(abs(p - 0.5) < 0.01 for p in nrfi_probs) and len(nrfi_probs) > 2:
        validation_errors.append("NRFI 概率单一（全部≈0.5），可能模型未加载")

if validation_errors:
    print("❌ 验证失败:")
    for e in validation_errors:
        print(e)
    if diagnostics['pipeline_errors']:
        print("最近 pipeline errors:")
        for e in diagnostics['pipeline_errors']:
            print(f"  {e}")
    with open('report/validation_errors.txt', 'w') as f:
        f.write('\n'.join(validation_errors))
    sys.exit(1)
else:
    print("✅ 预测输出通过自动验证")
