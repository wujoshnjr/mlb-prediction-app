from __future__ import annotations

from prediction import build_data_quality_status


def _quality(
    *,
    lineup_status: str = "confirmed",
    pitcher_status: str = "confirmed",
    starter_confirmation_pending: bool = False,
    odds_ok: bool = True,
    statcast_available: bool = True,
) -> dict:
    return build_data_quality_status(
        schedule_fetch_ok=True,
        odds_quality_status="OK" if odds_ok else "MISSING",
        odds_are_usable=odds_ok,
        daily_context_summary={
            "lineup_status": lineup_status,
            "pitcher_status": pitcher_status,
            "starter_confidence_status": "known",
            "starter_confirmation_pending": starter_confirmation_pending,
        },
        weather_context_row={"weather_source_status": "forecast_ok"},
        pitcher_advanced_row={"pitcher_advanced_source_status": "ok"},
        context_bridge_row={"context_bridge_source_status": "ok"},
        team_form_row={"team_form_source_status": "ok"},
        savant_top3_available=statcast_available,
    )


def test_bet_blocked_by_lineup_pending() -> None:
    result = _quality(lineup_status="pending")
    assert result["bet_allowed"] is False
    assert result["data_quality_grade"] == "C"
    assert "confirmed_lineup" in result["missing_important_sources"]


def test_bet_blocked_by_pitcher_not_confirmed() -> None:
    result = _quality(
        pitcher_status="high_confidence_probable",
        starter_confirmation_pending=True,
    )
    assert result["bet_allowed"] is False
    assert result["data_quality_grade"] == "C"
    assert "confirmed_starter" in result["missing_important_sources"]


def test_bet_blocked_by_odds_not_ok() -> None:
    result = _quality(odds_ok=False)
    assert result["bet_allowed"] is False
    assert result["missing_critical_sources"]


def test_bet_blocked_by_statcast_missing() -> None:
    result = _quality(statcast_available=False)
    assert result["bet_allowed"] is False
    assert result["missing_critical_sources"]


def test_bet_allowed_when_core_sources_confirmed() -> None:
    result = _quality()
    assert result["bet_allowed"] is True
    assert result["data_quality_grade"] == "A"
