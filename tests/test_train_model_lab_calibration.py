import numpy as np
import pandas as pd

import scripts.train_model_lab as train_model_lab


def _balanced_binary_labels(count: int, offset: int = 0) -> np.ndarray:
    return np.asarray([(index + offset) % 2 for index in range(count)], dtype=int)


def make_split(
    train_rows: int = 80,
    calibration_rows: int = 20,
    validation_rows: int = 20,
    n_features: int = 4,
    calibration_class_balance: str = "balanced",
    random_state: int = 42,
) -> dict:
    rng = np.random.RandomState(random_state)

    X_train = rng.normal(size=(train_rows, n_features))
    X_calibration = rng.normal(size=(calibration_rows, n_features))
    X_validation = rng.normal(size=(validation_rows, n_features))

    y_train = _balanced_binary_labels(train_rows)
    y_validation = _balanced_binary_labels(validation_rows, offset=1)

    if calibration_class_balance == "one_class":
        y_calibration = np.ones(calibration_rows, dtype=int)
    else:
        y_calibration = _balanced_binary_labels(calibration_rows)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_calibration": X_calibration,
        "y_calibration": y_calibration,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "train_count": train_rows,
        "calibration_count": calibration_rows,
        "validation_count": validation_rows,
    }


def test_logistic_uses_sigmoid_calibration():
    split = make_split(calibration_class_balance="balanced")
    features_used = ["f1", "f2", "f3", "f4"]

    result, model = train_model_lab._fit_logistic(
        split=split,
        features_used=features_used,
        experimental_features_used=[],
        sample_count=120,
        market_metrics=None,
    )

    assert result["trained"] is True
    assert result["skipped"] is False
    assert result["calibration_method"] == "sigmoid"
    assert result["calibration_used"] is True
    assert result["calibration_required"] is True
    assert result["calibration_count"] == 20

    assert model is not None
    assert hasattr(model, "predict_proba")

    probability = model.predict_proba(split["X_validation"])[:, 1]
    assert np.all(np.isfinite(probability))
    assert np.all(probability >= 0)
    assert np.all(probability <= 1)


def test_logistic_skips_when_calibration_has_one_class():
    split = make_split(calibration_class_balance="one_class")
    features_used = ["f1", "f2", "f3", "f4"]

    result, model = train_model_lab._fit_logistic(
        split=split,
        features_used=features_used,
        experimental_features_used=[],
        sample_count=120,
        market_metrics=None,
    )

    assert result["trained"] is False
    assert result["skipped"] is True
    assert result["calibration_used"] is False
    assert result["calibration_method"] == "sigmoid"
    assert "calibration set contains only one target class" in result["skip_reason"]
    assert model is None


def test_model_lab_build_report_uses_finalized_snapshot_outcomes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        train_model_lab,
        "MODEL_FEATURES",
        ["f1", "f2", "f3", "f4"],
    )

    def skipped_model_result(model_name: str):
        result = train_model_lab._base_model_result(model_name)
        result["trained"] = False
        result["skipped"] = True
        result["skip_reason"] = "skipped in unit test"
        result["promotion_blockers"] = [result["skip_reason"]]
        return result, None

    monkeypatch.setattr(
        train_model_lab,
        "_fit_lightgbm_classifier",
        lambda *args, **kwargs: skipped_model_result("lightgbm_classifier"),
    )
    monkeypatch.setattr(
        train_model_lab,
        "_fit_xgboost_classifier",
        lambda *args, **kwargs: skipped_model_result("xgboost_classifier"),
    )
    monkeypatch.setattr(
        train_model_lab,
        "_fit_lightgbm_market_residual",
        lambda *args, **kwargs: skipped_model_result("lightgbm_market_residual"),
    )

    snapshot_path = tmp_path / "prediction_snapshots.csv"
    outcome_path = tmp_path / "finalized_snapshot_outcomes.csv"
    report_path = tmp_path / "report" / "model_lab_report.json"
    artifact_dir = tmp_path / "artifacts"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 120
    game_ids = [str(index) for index in range(sample_count)]
    rng = np.random.RandomState(123)

    snapshots = pd.DataFrame(
        {
            "game_id": game_ids,
            "snapshot_created_at": pd.date_range(
                "2025-01-01",
                periods=sample_count,
                freq="h",
            ),
            "pipeline_version": train_model_lab.PIPELINE_VERSION,
            "snapshot_valid": True,
            "f1": rng.normal(size=sample_count),
            "f2": rng.normal(size=sample_count),
            "f3": rng.normal(size=sample_count),
            "f4": rng.normal(size=sample_count),
        }
    )
    snapshots.to_csv(snapshot_path, index=False)

    outcomes = pd.DataFrame(
        {
            "game_id": game_ids,
            "home_win": _balanced_binary_labels(sample_count),
        }
    )
    outcomes.to_csv(outcome_path, index=False)

    report = train_model_lab.build_report(
        snapshot_path=snapshot_path,
        finalized_path=outcome_path,
        report_path=report_path,
        artifact_dir=artifact_dir,
    )

    assert report["sample_count"] == sample_count
    assert report["train_count"] > 0
    assert report["calibration_count"] > 0
    assert report["validation_count"] > 0

    logistic = next(
        model
        for model in report["models"]
        if model["model_name"] == "logistic_baseline"
    )

    assert logistic["trained"] is True
    assert logistic["skipped"] is False
    assert logistic["calibration_method"] == "sigmoid"
    assert logistic["calibration_used"] is True
    assert logistic["calibration_required"] is True
    assert logistic["calibration_count"] > 0

    assert report_path.exists()
