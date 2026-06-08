from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.train_shadow_ensemble_stack import build_report


def test_missing_walk_forward_predictions_skipped(tmp_path: Path) -> None:
    report = build_report(
        predictions_path=tmp_path / "missing.csv",
        report_path=tmp_path / "report" / "shadow_ensemble_stack_report.json",
        artifact_path=tmp_path / "data" / "model_lab" / "shadow_ensemble_stack.pkl",
    )

    assert report["skipped"] is True
    assert report["promotion_eligible"] is False


def test_shadow_ensemble_report_generated(tmp_path: Path) -> None:
    rows = []
    for i in range(100):
        for model in ["logistic_baseline", "xgboost_classifier", "market_residual_model"]:
            rows.append(
                {
                    "game_id": str(1000 + i),
                    "model_name": model,
                    "predicted_prob": 0.45 + ((i % 10) / 100),
                    "actual_home_win": int(i % 2),
                }
            )

    pp = tmp_path / "walk_forward_predictions.csv"
    pd.DataFrame(rows).to_csv(pp, index=False)

    report = build_report(
        predictions_path=pp,
        report_path=tmp_path / "report" / "shadow_ensemble_stack_report.json",
        artifact_path=tmp_path / "data" / "model_lab" / "shadow_ensemble_stack.pkl",
    )

    assert report["skipped"] is False
    assert report["sample_count"] == 100
    assert report["promotion_eligible"] is False
    assert "sample_count below promotion threshold" in " ".join(report["promotion_blockers"])
