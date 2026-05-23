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
nrfi_probs = [p['nrfi_prob'] for p in preds]

errors = []
if home_probs:
    mean_home = np.mean(home_probs)
    if mean_home > 0.58 or mean_home < 0.45:
        errors.append(f"Avg home prob = {mean_home:.3f}, 偏离正常区间")

if over_probs:
    mean_over = np.mean(over_probs)
    if mean_over > 0.9:
        errors.append(f"Mean over prob = {mean_over:.3f}, 仿真可能崩溃")

if nrfi_probs:
    unique_nrfi = len(set(round(p, 3) for p in nrfi_probs))
    if unique_nrfi <= 2:
        errors.append("NRFI 概率单一，可能模型未加载")

if errors:
    with open('report/validation_errors.txt', 'w') as f:
        f.write('\n'.join(errors))
    print("❌ 验证失败:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("✅ 预测输出通过自动验证")
