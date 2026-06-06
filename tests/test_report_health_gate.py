from __future__ import annotations

from scripts.report_health_gate import (
    _check_feature_availability,
    _contains_literal_nan,
)


def test_contains_literal_nan_returns_paths() -> None:
    assert _contains_literal_nan({"a": "nan"}) == ["a"]
    assert _contains_literal_nan({"a": "NaN"}) == ["a"]
    assert _contains_literal_nan({"a": ""}) == []
    assert _contains_literal_nan({"a": {"b": "nan"}}) == ["a.b"]


def test_tracking_only_high_risk_is_error() -> None:
    errors: list[str] = []
    warnings: list[str] = []

    _check_feature_availability(
        {
            "non_blocking_features": [],
            "high_risk_features": ["dynamic_pythag_diff"],
            "group_summary": {},
        },
        errors,
        warnings,
    )

    assert errors
    assert any("dynamic_pythag_diff" in error for error in errors)
