from **future** import annotations

import json
from pathlib import Path
from typing import Any

from scripts.away_guardrail_impact_report import compute_report, safe_json_dump

def make_prediction(
game_id: str,
side: str,
recommendation_status: str,
applied: bool,
reasons: list[str] | None = None,
) -> dict[str, Any]:
return {
"game_id": game_id,
"game_date": "2026-06-10",
"away_team": "Away",
"home_team": "Home",
"moneyline_selected_side": side,
"moneyline_selected_edge": 0.04,
"away_selected_probability": 0.55,
"market_no_vig_home_prob": 0.52,
"recommendation_status": recommendation_status,
"away_guardrail_applied": applied,
"away_guardrail_status": "tracking_only" if applied else "pass",
"away_guardrail_reasons": reasons or [],
}

def test_missing_prediction_report_partial() -> None:
report = compute_report(None, None, prediction_error="file_missing")

```
assert report["status"] == "partial"
assert report["errors"]
assert report["summary"]["prediction_count"] == 0
```

def test_counts_away_candidates() -> None:
prediction_report = {
"predictions": [
make_prediction("1", "away", "PAPER_BET", False),
make_prediction("2", "home", "PAPER_BET", False),
]
}

```
report = compute_report(prediction_report)

assert report["summary"]["prediction_count"] == 2
assert report["summary"]["away_candidate_count"] == 1
assert report["summary"]["home_candidate_count"] == 1
```

def test_counts_guardrail_applied() -> None:
prediction_report = {
"predictions": [
make_prediction(
"1",
"away",
"TRACKING_ONLY",
True,
["away_mid_edge_underperforming_bucket"],
),
make_prediction("2", "away", "PAPER_BET", False),
]
}

```
report = compute_report(prediction_report)

assert report["summary"]["away_candidate_count"] == 2
assert report["summary"]["away_guardrail_applied_count"] == 1
assert report["summary"]["away_guardrail_applied_rate"] == 0.5
```

def test_retained_away_paper_signal_count() -> None:
prediction_report = {
"predictions": [
make_prediction("1", "away", "PAPER_BET", False),
make_prediction("2", "away", "TRACKING_ONLY", True),
]
}

```
report = compute_report(prediction_report)

assert report["summary"]["retained_away_paper_signal_count"] == 1
assert report["summary"]["downgraded_away_tracking_only_count"] == 1
```

def test_reason_counts() -> None:
prediction_report = {
"predictions": [
make_prediction("1", "away", "TRACKING_ONLY", True, ["a", "b"]),
make_prediction("2", "away", "TRACKING_ONLY", True, ["a"]),
]
}

```
report = compute_report(prediction_report)

assert report["reason_counts"]["a"] == 2
assert report["reason_counts"]["b"] == 1
```

def test_downgraded_examples_capped_at_10() -> None:
prediction_report = {
"predictions": [
make_prediction(str(index), "away", "TRACKING_ONLY", True, ["a"])
for index in range(20)
]
}

```
report = compute_report(prediction_report)

assert len(report["downgraded_examples"]) == 10
```

def test_json_serializable(tmp_path: Path) -> None:
prediction_report = {
"predictions": [
make_prediction("1", "away", "TRACKING_ONLY", True, ["a"])
]
}

```
report = compute_report(prediction_report)
output_path = tmp_path / "report.json"

safe_json_dump(report, output_path)

loaded = json.loads(output_path.read_text(encoding="utf-8"))

assert loaded["summary"]["away_candidate_count"] == 1
```

def test_safety_flags_false() -> None:
report = compute_report({"predictions": []})

```
assert report["live_betting_allowed"] is False
assert report["automated_wagering_allowed"] is False
assert report["production_model_replacement_allowed"] is False
```

def test_non_away_ignored_for_away_counts() -> None:
prediction_report = {
"predictions": [
make_prediction("1", "home", "PAPER_BET", False),
]
}

```
report = compute_report(prediction_report)

assert report["summary"]["away_candidate_count"] == 0
assert report["summary"]["home_candidate_count"] == 1
```

def test_diagnostic_snapshot() -> None:
prediction_report = {"predictions": []}
diagnostic_report = {
"official_accuracy": {
"away_picks": {
"sample_count": 110,
"accuracy": 0.4909,
}
},
"sample_summary": {
"away_pick_rate": 0.5628,
},
"recommended_guardrails": ["raise away threshold"],
}

```
report = compute_report(prediction_report, diagnostic_report)

assert report["diagnostic_snapshot"]["away_pick_sample_count"] == 110
assert report["diagnostic_snapshot"]["away_pick_accuracy"] == 0.4909
assert report["diagnostic_snapshot"]["away_pick_rate"] == 0.5628
```
