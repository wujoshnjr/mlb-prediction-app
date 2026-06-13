import numpy as np
import pandas as pd

import scripts.run_walk_forward_validation as rwv


def _balanced_binary_labels(count: int, offset: int = 0) -> np.ndarray:
    return np.asarray([(index + offset) % 2 for index in range(count)], dtype=int)


def test_fit_predict_logistic_uses_calibration():
    rng = np.random.RandomState(42)

    X_train = rng.normal(size=(100, 4))
    y_train = _balanced_binary_labels(100)
    X_valid = rng.normal(size=(10, 4))

    probability, warnings, calibration_used = rwv._fit_predict(
        "logistic_baseline",
        X_train,
        y_train,
        X_valid,
    )

    assert probability is not None
    assert len(probability) == 10
    assert calibration_used is True
    assert isinstance(warnings, list)
    assert np.all(np.isfinite(probability))
    assert np.all(probability >= 0.01)
    assert np.all(probability <= 0.99)


def test_fit_predict_skips_when_inner_calibration_one_class():
    rng = np.random.RandomState(43)

    X_train = rng.normal(size=(100, 4))
    y_train = np.asarray([index % 2 for index in range(85)] + [1] * 15, dtype=int)
    X_valid = rng.normal(size=(10, 4))

    probability, warnings, calibration_used = rwv._fit_predict(
        "logistic_baseline",
        X_train,
        y_train,
        X_valid,
    )

    assert probability is None
    assert calibration_used is False
    assert any(
        "inner calibration target contains one class" in warning
        for warning in warnings
    )


def test_build_report_outputs_calibration_used_by_model(tmp_path, monkeypatch):
    monkeypatch.setattr(rwv, "MODEL_FEATURES", ["f1", "f2", "f3", "f4"])

    original_fit_predict = rwv._fit_predict

    def fake_fit_predict(model_name, X_train, y_train, X_valid):
        if model_name == "logistic_baseline":
            return original_fit_predict(model_name, X_train, y_train, X_valid)

        return None, ["skipped in unit test"], False

    monkeypatch.setattr(rwv, "_fit_predict", fake_fit_predict)

    snapshot_path = tmp_path / "prediction_snapshots.csv"
    outcome_path = tmp_path / "finalized_snapshot_outcomes.csv"
    report_path = tmp_path / "report" / "walk_forward_validation_report.json"
    predictions_path = tmp_path / "data" / "walk_forward_predictionsprediction_snapshots.csv"
    outcome_path = tmp_path / "finalized_snapshot.csv"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    sample_count = 140
    game_ids = [str(index) for index in range(sample_count)]
    rng = np.random.RandomState(44)

    snapshots = pd.DataFrame(
        {
            "game_id": game_ids,
            "snapshot_created_at": pd.date_range(
                "2025-01-01",
                periods=sample_count,
                freq="h",
            ),
            "pipeline_version": rwv.PIPELINE_VERSION,
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

    report = rwv.build_report(
        snapshot_path=snapshot_path,
        finalized_path=outcome_path,
        report_path=report_path,
        predictions_path=predictions_path,
        minimum_train_samples=80,
        validation_window_size=10,
        step_size=10,
    )

    assert report["skipped"] is False
    assert report["calibration_method_by_model"]["logistic_baseline"] == "sigmoid"
    assert report["calibration_used_by_model"]["logistic_baseline"] is True
    assert report["calibration_used_by_model"]["market_no_vig_baseline"] is False
    assert report["calibration_used_by_model"]["market_residual_model"] is False
    assert "per_model_ready" in report
    assert "minimum_required_oos_predictions" in report
    assert report["minimum_required_oos_predictions"] == 300
    assert predictions_path.exists()
    assert report_path.exists()


def test_per_model_oos_readiness_does_not_sum_models():
    predictions = pd.DataFrame(
        {
            "model_name": ["logistic_baseline"] * 5 + ["xgboost_classifier"] * 5,
            "game_id": [101, 102, 103, 104, 105] * 2,
        }
    )

    counts = rwv._model_oos_counts(predictions)
    unique_games = rwv._unique_oos_game_count(predictions)

    assert counts.get("logistic_baseline") == 5
    assert counts.get("xgboost_classifier") == 5
    assert unique_games == 5
