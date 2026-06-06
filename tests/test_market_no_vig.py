from __future__ import annotations

import pytest

from scripts.baseline_comparison_report import _no_vig_home_probability


def test_no_vig_equal_odds_home_probability() -> None:
    assert _no_vig_home_probability(2.0, 2.0) == pytest.approx(0.5)


def test_no_vig_probabilities_sum_to_one() -> None:
    home = _no_vig_home_probability(1.8, 2.2)
    away = _no_vig_home_probability(2.2, 1.8)
    assert home is not None
    assert away is not None
    assert home + away == pytest.approx(1.0)


def test_no_vig_invalid_odds_return_none() -> None:
    assert _no_vig_home_probability(1.0, 2.0) is None
    assert _no_vig_home_probability(0.5, 2.0) is None
