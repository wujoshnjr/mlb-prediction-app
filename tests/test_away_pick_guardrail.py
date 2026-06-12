from **future** import annotations

from prediction import evaluate_away_pick_guardrail

def ctx(
pitcher_status: str = "confirmed",
lineup_status: str = "confirmed",
bullpen_status: str = "available",
) -> dict[str, str]:
return {
"pitcher_status": pitcher_status,
"lineup_status": lineup_status,
"bullpen_status": bullpen_status,
}

def test_home_side_not_applicable() -> None:
result = evaluate_away_pick_guardrail(
selected_side="home",
selected_edge=0.10,
selected_probability=0.60,
market_home_probability=0.55,
daily_context_summary=ctx(),
)

```
assert result["away_guardrail_status"] == "not_applicable"
assert result["away_guardrail_applied"] is False
```

def test_away_three_to_five_edge_tracking_only() -> None:
result = evaluate_away_pick_guardrail(
selected_side="away",
selected_edge=0.04,
selected_probability=0.62,
market_home_probability=0.45,
daily_context_summary=ctx(),
)

```
assert result["away_guardrail_status"] == "tracking_only"
assert result["away_guardrail_applied"] is True
assert "away_mid_edge_underperforming_bucket" in result["away_guardrail_reasons"]
```

def test_away_five_to_eight_edge_tracking_only() -> None:
result = evaluate_away_pick_guardrail(
selected_side="away",
selected_edge=0.065,
selected_probability=0.62,
market_home_probability=0.45,
daily_context_summary=ctx(),
)

```
assert result["away_guardrail_status"] == "tracking_only"
assert "away_mid_edge_underperforming_bucket" in result["away_guardrail_reasons"]
```

def test_away_large_edge_alone_passes() -> None:
result = evaluate_away_pick_guardrail(
selected_side="away",
selected_edge=0.09,
selected_probability=0.62,
market_home_probability=0.45,
daily_context_summary=ctx(),
)

```
assert result["away_guardrail_status"] == "pass"
assert result["away_guardrail_applied"] is False
```

def test_slight_home_market_tracking_only() -> None:
result = evaluate_away_pick_guardrail(
selected_side="away",
selected_edge=0.10,
selected_probability=0.62,
market_home_probability=0.52,
daily_context_summary=ctx(),
)

```
assert result["away_guardrail_status"] == "tracking_only"
assert (
    "away_slight_home_market_underperforming_bucket"
    in result["away_guardrail_reasons"]
)
```

def test_home_market_low_away_probability_tracking_only() -> None:
result = evaluate_away_pick_guardrail(
selected_side="away",
selected_edge=0.10,
selected_probability=0.55,
market_home_probability=0.60,
daily_context_summary=ctx(),
)

```
assert result["away_guardrail_status"] == "tracking_only"
assert (
    "away_low_selected_probability_vs_home_market"
    in result["away_guardrail_reasons"]
)
```

def test_context_unconfirmed_note_only() -> None:
result = evaluate_away_pick_guardrail(
selected_side="away",
selected_edge=0.10,
selected_probability=0.62,
market_home_probability=0.45,
daily_context_summary=ctx(pitcher_status="unknown"),
)

```
assert result["away_guardrail_status"] == "pass"
assert result["away_guardrail_applied"] is False
assert "away_context_unconfirmed" in result["away_guardrail_notes"]
```

def test_multiple_reasons_can_exist() -> None:
result = evaluate_away_pick_guardrail(
selected_side="away",
selected_edge=0.04,
selected_probability=0.55,
market_home_probability=0.52,
daily_context_summary=ctx(),
)

```
reasons = result["away_guardrail_reasons"]

assert result["away_guardrail_status"] == "tracking_only"
assert "away_mid_edge_underperforming_bucket" in reasons
assert "away_slight_home_market_underperforming_bucket" in reasons
assert "away_low_selected_probability_vs_home_market" in reasons
```

def test_none_values_do_not_crash() -> None:
result = evaluate_away_pick_guardrail(
selected_side="away",
selected_edge=None,
selected_probability=None,
market_home_probability=None,
daily_context_summary=None,
)

```
assert result["away_guardrail_status"] == "pass"
assert result["away_guardrail_applied"] is False
assert result["away_selected_edge"] is None
```

def test_away_market_probability_inverse() -> None:
result = evaluate_away_pick_guardrail(
selected_side="away",
selected_edge=0.10,
selected_probability=0.62,
market_home_probability=0.40,
daily_context_summary=ctx(),
)

```
assert result["away_market_probability"] == 0.6
```
