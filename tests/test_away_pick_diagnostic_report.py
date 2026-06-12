from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.away_pick_diagnostic_report import compute_report, safe_json_dump


def make_predictions(
    game_ids: list[object],
    dates: list[str],
    home_probs: list[float],
    sides: list[str],
    market_home_probs: list[float] | None = None,
    model_edge_home: list[float] | None = None,
    pitcher_available: list[bool] | None = None,
    bullpen_available: list[bool] | None = None,
    lineup_confirmed: list[bool] | None = None,
) -> pd.DataFrame:
    rows: dict[str, object] = {
        "game_id": game_ids,
        "game_date": dates,
        "pipeline_version": ["baseline_v2_clean"] * len(game_ids),
        "snapshot_valid": ["true"] * len(game_ids),
        "predicted_home_win_pct": home_probs,
        "moneyline_selected_side": sides,
    }

    if market_home_probs is not None:
        rows["market_no_vig_home_prob"] = market_home_probs
    if model_edge_home is not None:
        rows["model_edge_home"] = model_edge_home
    if pitcher_available is not None:
        rows["pitcher_advanced_available"] = pitcher_available
    if bullpen_available is not None:
        rows["bullpen_context_available"] = bullpen_available
    if lineup_confirmed is not None:
        rows["lineup_confirmed"] = lineup_confirmed

    return pd.DataFrame(rows)


def make_outcomes(game_ids: list[object], home_wins: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": game_ids,
            "home_win": home_wins,
        }
    )


def test_away_pick_accuracy_correct() -> None:
    predictions = make_predictions(
        game_ids=["g1"],
        dates=["2026-06-10"],
        home_probs=[0.40],
        sides=["away"],
    )
    outcomes = make_outcomes(["g1"], [0])

    report = compute_report(predictions, outcomes)

    assert report["official_accuracy"]["away_picks"]["sample_count"] == 1
    assert report["official_accuracy"]["away_picks"]["correct"] == 1
    assert report["official_accuracy"]["away_picks"]["accuracy"] == 1.0
    assert report["live_betting_allowed"] is False
    assert report["automated_wagering_allowed"] is False
    assert report["production_model_replacement_allowed"] is False


def test_home_and_away_pick_accuracy_are_separated() -> None:
    predictions = make_predictions(
        game_ids=["g1", "g2"],
        dates=["2026-06-10", "2026-06-10"],
        home_probs=[0.60, 0.40],
        sides=["home", "away"],
    )
    outcomes = make_outcomes(["g1", "g2"], [1, 0])

    report = compute_report(predictions, outcomes)

    assert report["official_accuracy"]["all_picks"]["sample_count"] == 2
    assert report["official_accuracy"]["home_picks"]["sample_count"] == 1
    assert report["official_accuracy"]["away_picks"]["sample_count"] == 1
    assert report["official_accuracy"]["home_picks"]["correct"] == 1
    assert report["official_accuracy"]["away_picks"]["correct"] == 1


def test_pending_predictions_are_excluded_from_accuracy_denominator() -> None:
    predictions = make_predictions(
        game_ids=["g1", "g2"],
        dates=["2026-06-10", "2026-06-10"],
        home_probs=[0.40, 0.35],
        sides=["away", "away"],
    )
    outcomes = make_outcomes(["g1"], [0])

    report = compute_report(predictions, outcomes)

    assert report["sample_summary"]["away_pick_count"] == 2
    assert report["sample_summary"]["away_pick_settled_count"] == 1
    assert report["sample_summary"]["away_pick_pending_count"] == 1
    assert report["official_accuracy"]["away_picks"]["sample_count"] == 1


def test_away_favorite_and_away_underdog_segments() -> None:
    predictions = make_predictions(
        game_ids=["g1", "g2"],
        dates=["2026-06-10", "2026-06-10"],
        home_probs=[0.40, 0.40],
        sides=["away", "away"],
        market_home_probs=[0.45, 0.60],
    )
    outcomes = make_outcomes(["g1", "g2"], [0, 1])

    report = compute_report(predictions, outcomes)

    assert report["away_segments"]["away_favorites"]["sample_count"] == 1
    assert report["away_segments"]["away_favorites"]["correct"] == 1
    assert report["away_segments"]["away_underdogs"]["sample_count"] == 1
    assert report["away_segments"]["away_underdogs"]["correct"] == 0


def test_edge_buckets_are_calculated_for_away_selected_edge() -> None:
    predictions = make_predictions(
        game_ids=["g1", "g2", "g3", "g4"],
        dates=["2026-06-10"] * 4,
        home_probs=[0.49, 0.48, 0.45, 0.40],
        sides=["away", "away", "away", "away"],
        model_edge_home=[-0.02, -0.04, -0.06, -0.10],
    )
    outcomes = make_outcomes(["g1", "g2", "g3", "g4"], [0, 0, 0, 0])

    report = compute_report(predictions, outcomes)

    buckets = {bucket["label"]: bucket for bucket in report["away_by_edge_bucket"]}

    assert buckets["<3%"]["sample_count"] == 1
    assert buckets["3-5%"]["sample_count"] == 1
    assert buckets["5-8%"]["sample_count"] == 1
    assert buckets[">=8%"]["sample_count"] == 1


def test_missing_files_partial_no_crash() -> None:
    report = compute_report(None, None)

    assert report["status"] == "partial"
    assert report["metadata"]["status"] == "partial"
    assert report["errors"]
    assert report["official_accuracy"]["away_picks"]["sample_count"] == 0


def test_safety_flags_always_false() -> None:
    predictions = make_predictions(
        game_ids=["g1"],
        dates=["2026-06-10"],
        home_probs=[0.40],
        sides=["away"],
    )
    outcomes = make_outcomes(["g1"], [0])

    report = compute_report(predictions, outcomes)

    assert report["live_betting_allowed"] is False
    assert report["automated_wagering_allowed"] is False
    assert report["production_model_replacement_allowed"] is False
    assert report["metadata"]["live_betting_allowed"] is False
    assert report["metadata"]["automated_wagering_allowed"] is False
    assert report["metadata"]["production_model_replacement_allowed"] is False
    assert report["betting_mode"] == "paper_research"


def test_json_serializable_no_nan(tmp_path: Path) -> None:
    predictions = make_predictions(
        game_ids=["g1"],
        dates=["2026-06-10"],
        home_probs=[0.40],
        sides=["away"],
    )
    outcomes = make_outcomes(["g1"], [1])

    report = compute_report(predictions, outcomes)
    output_path = tmp_path / "away_pick_diagnostic_report.json"

    safe_json_dump(report, output_path)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded["official_accuracy"]["away_picks"]["sample_count"] == 1
    assert loaded["official_accuracy"]["away_picks"]["accuracy"] == 0.0


def test_recommendation_when_away_underdogs_are_weak() -> None:
    game_ids = [f"g{i}" for i in range(20)]
    predictions = make_predictions(
        game_ids=game_ids,
        dates=["2026-06-10"] * 20,
        home_probs=[0.40] * 20,
        sides=["away"] * 20,
        market_home_probs=[0.60] * 20,
        model_edge_home=[-0.05] * 20,
    )
    outcomes = make_outcomes(game_ids, [1] * 20)

    report = compute_report(predictions, outcomes)

    assert report["away_segments"]["away_underdogs"]["sample_count"] == 20
    assert report["away_segments"]["away_underdogs"]["accuracy"] == 0.0
    assert any(
        "Away underdogs are weak" in item
        for item in report["recommended_guardrails"]
    )


def test_game_id_string_int_join() -> None:
    predictions = make_predictions(
        game_ids=["823295"],
        dates=["2026-06-10"],
        home_probs=[0.40],
        sides=["away"],
    )
    outcomes = make_outcomes([823295], [0])

    report = compute_report(predictions, outcomes)

    assert report["official_accuracy"]["away_picks"]["sample_count"] == 1
    assert report["official_accuracy"]["away_picks"]["correct"] == 1
