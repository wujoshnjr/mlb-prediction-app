from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.daily_model_accuracy_report import compute_report, safe_json_dump


def make_predictions(
    game_ids: list[str],
    dates: list[str],
    home_probs: list[float] | None = None,
    selected_side: list[str] | None = None,
    recommendation_status: list[str] | None = None,
    market_home_probs: list[float] | None = None,
) -> pd.DataFrame:
    rows: dict[str, object] = {
        "game_id": game_ids,
        "game_date": dates,
        "pipeline_version": ["baseline_v2_clean"] * len(game_ids),
        "snapshot_valid": ["true"] * len(game_ids),
    }

    if home_probs is not None:
        rows["predicted_home_win_pct"] = home_probs
    if selected_side is not None:
        rows["moneyline_selected_side"] = selected_side
    if recommendation_status is not None:
        rows["recommendation_status"] = recommendation_status
    if market_home_probs is not None:
        rows["market_no_vig_home_prob"] = market_home_probs

    return pd.DataFrame(rows)


def make_outcomes(game_ids: list[str], home_wins: list[int | float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": game_ids,
            "home_win": home_wins,
        }
    )


def test_home_pick_correct() -> None:
    predictions = make_predictions(
        ["g1"],
        ["2026-06-10"],
        home_probs=[0.7],
    )
    outcomes = make_outcomes(["g1"], [1])

    report = compute_report(predictions, outcomes)

    assert report["official_accuracy"]["sample_count"] == 1
    assert report["official_accuracy"]["correct"] == 1
    assert report["official_accuracy"]["accuracy"] == 1.0
    assert report["live_betting_allowed"] is False


def test_away_pick_correct() -> None:
    predictions = make_predictions(
        ["g2"],
        ["2026-06-10"],
        home_probs=[0.3],
    )
    outcomes = make_outcomes(["g2"], [0])

    report = compute_report(predictions, outcomes)

    assert report["official_accuracy"]["sample_count"] == 1
    assert report["official_accuracy"]["correct"] == 1
    assert report["official_accuracy"]["accuracy"] == 1.0


def test_pending_outcome_not_in_denominator() -> None:
    predictions = make_predictions(
        ["g3", "g4"],
        ["2026-06-10", "2026-06-10"],
        home_probs=[0.6, 0.7],
    )
    outcomes = make_outcomes(["g3"], [1])

    report = compute_report(predictions, outcomes)

    assert report["official_accuracy"]["sample_count"] == 1
    assert report["official_accuracy"]["correct"] == 1
    assert report["pending_predictions"]["count"] == 1


def test_missing_or_invalid_home_win_not_in_denominator() -> None:
    predictions = make_predictions(
        ["g5"],
        ["2026-06-10"],
        home_probs=[0.6],
    )
    outcomes = make_outcomes(["g5"], [float("nan")])

    report = compute_report(predictions, outcomes)

    assert report["official_accuracy"]["sample_count"] == 0
    assert report["official_accuracy"]["accuracy"] is None
    assert report["pending_predictions"]["count"] == 1


def test_clv_does_not_affect_official_accuracy() -> None:
    predictions = make_predictions(
        ["g6"],
        ["2026-06-10"],
        home_probs=[0.55],
    )
    outcomes = make_outcomes(["g6"], [1])
    clv = {
        "clv_summary": {
            "avg_clv": -999.0,
            "positive_clv_rate": 0.0,
            "evaluated_picks": 50,
        }
    }

    report = compute_report(predictions, outcomes, clv_data=clv)

    assert report["official_accuracy"]["sample_count"] == 1
    assert report["official_accuracy"]["correct"] == 1
    assert report["official_accuracy"]["accuracy"] == 1.0
    assert report["clv_metrics"]["available"] is True
    assert report["clv_metrics"]["avg_clv"] == -999.0
    assert "not win/loss accuracy" in report["clv_metrics"]["note"]


def test_missing_files_partial_no_crash() -> None:
    report = compute_report(None, None)

    assert report["status"] == "partial"
    assert report["official_accuracy"]["sample_count"] == 0
    assert report["errors"]

    predictions = make_predictions(
        ["g7"],
        ["2026-06-10"],
        home_probs=[0.5],
    )
    report_with_missing_outcomes = compute_report(predictions, None)

    assert report_with_missing_outcomes["status"] == "partial"
    assert report_with_missing_outcomes["official_accuracy"]["sample_count"] == 0
    assert report_with_missing_outcomes["errors"]


def test_safety_flags_always_false() -> None:
    predictions = make_predictions(
        ["g8"],
        ["2026-06-10"],
        home_probs=[0.5],
    )
    outcomes = make_outcomes(["g8"], [1])

    report = compute_report(predictions, outcomes)

    assert report["live_betting_allowed"] is False
    assert report["automated_wagering_allowed"] is False
    assert report["production_model_replacement_allowed"] is False


def test_json_serializable_no_nan(tmp_path: Path) -> None:
    predictions = make_predictions(
        ["g9"],
        ["2026-06-10"],
        home_probs=[0.5],
    )
    outcomes = make_outcomes(["g9"], [0])

    report = compute_report(predictions, outcomes)

    output_path = tmp_path / "daily_model_accuracy_report.json"
    safe_json_dump(report, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded["official_accuracy"]["sample_count"] == 1
    assert loaded["official_accuracy"]["accuracy"] == 0.0


def test_game_id_normalization_allows_string_int_join() -> None:
    predictions = make_predictions(
        ["823295"],
        ["2026-06-10"],
        home_probs=[0.61],
    )
    outcomes = make_outcomes([823295], [1])

    report = compute_report(predictions, outcomes)

    assert report["official_accuracy"]["sample_count"] == 1
    assert report["official_accuracy"]["correct"] == 1


def test_paper_signal_from_recommendation_status() -> None:
    predictions = make_predictions(
        ["g10", "g11"],
        ["2026-06-10", "2026-06-10"],
        home_probs=[0.6, 0.4],
        recommendation_status=["PAPER_BET", "TRACKING_ONLY"],
    )
    outcomes = make_outcomes(["g10", "g11"], [1, 0])

    report = compute_report(predictions, outcomes)

    assert report["slices"]["paper_signals"]["sample_count"] == 1
    assert report["slices"]["paper_signals"]["correct"] == 1
    assert report["slices"]["tracking_only"]["sample_count"] == 1
    assert report["slices"]["tracking_only"]["correct"] == 1


def test_favorite_underdog_slices_use_market_home_probability() -> None:
    predictions = make_predictions(
        ["g12", "g13"],
        ["2026-06-10", "2026-06-10"],
        home_probs=[0.6, 0.4],
        market_home_probs=[0.55, 0.70],
    )
    outcomes = make_outcomes(["g12", "g13"], [1, 0])

    report = compute_report(predictions, outcomes)

    assert report["slices"]["favorites"]["sample_count"] == 1
    assert report["slices"]["favorites"]["correct"] == 1
    assert report["slices"]["underdogs"]["sample_count"] == 1
    assert report["slices"]["underdogs"]["correct"] == 1
