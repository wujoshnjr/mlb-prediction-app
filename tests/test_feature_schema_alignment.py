from __future__ import annotations

from scripts.feature_schema import EXPECTED_FEATURES, MODEL_FEATURES, TRACKING_ONLY_FEATURES


def test_feature_lists_have_no_duplicates() -> None:
    assert len(EXPECTED_FEATURES) == len(set(EXPECTED_FEATURES))
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))
    assert len(TRACKING_ONLY_FEATURES) == len(set(TRACKING_ONLY_FEATURES))


def test_model_and_tracking_features_do_not_overlap() -> None:
    assert set(MODEL_FEATURES).isdisjoint(set(TRACKING_ONLY_FEATURES))


def test_model_features_are_subset_of_expected_features() -> None:
    assert set(MODEL_FEATURES).issubset(set(EXPECTED_FEATURES))


def test_tracking_features_are_subset_of_expected_features() -> None:
    assert set(TRACKING_ONLY_FEATURES).issubset(set(EXPECTED_FEATURES))
