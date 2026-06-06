from __future__ import annotations

import math

import pytest


def decimal_to_implied_probability(odds: float) -> float:
    if odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return 1.0 / odds


def clip_probability(probability: float, epsilon: float = 1e-12) -> float:
    return max(epsilon, min(1.0 - epsilon, float(probability)))


def test_decimal_odds_to_implied_probability() -> None:
    assert decimal_to_implied_probability(2.0) == pytest.approx(0.5)
    assert decimal_to_implied_probability(1.5) == pytest.approx(2.0 / 3.0)


def test_invalid_decimal_odds_raise() -> None:
    with pytest.raises(ValueError):
        decimal_to_implied_probability(1.0)

    with pytest.raises(ValueError):
        decimal_to_implied_probability(0.9)


def test_probability_clip_avoids_log_zero() -> None:
    assert 0.0 < clip_probability(0.0) < 1.0
    assert 0.0 < clip_probability(1.0) < 1.0

    p0 = clip_probability(0.0)
    p1 = clip_probability(1.0)
    assert math.isfinite(math.log(p0))
    assert math.isfinite(math.log(1.0 - p1))
