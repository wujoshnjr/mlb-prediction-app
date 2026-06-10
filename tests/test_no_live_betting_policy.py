from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


CHECK_PATHS = [
    Path("report/prediction.json"),
    Path("data/sample_state.json"),
    Path("data/training_status.json"),
    Path("data/model_artifact_status.json"),
    Path("report/product_experience_report.json"),
    Path("report/pipeline_manifest.json"),
]

FORBIDDEN_TRUE_KEYS = {
    "live_betting_allowed",
    "automated_wagering_allowed",
    "production_allowed",
    "production_model_replacement_allowed",
    "shadow_live_allowed",
}


def _walk_values(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)


@pytest.mark.parametrize("path", CHECK_PATHS)
def test_no_live_betting_policy(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path} does not exist")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pytest.skip(f"{path} is not valid JSON")

    if not isinstance(payload, dict):
        pytest.skip(f"{path} is not a JSON object")

    for obj in _walk_values(payload):
        if not isinstance(obj, dict):
            continue

        for key in FORBIDDEN_TRUE_KEYS:
            assert obj.get(key) is not True, f"{key} is true in {path}"

        if "stake_multiplier" in obj:
            assert obj.get("signal_type") in {
                "paper_signal",
                "paper_research_signal",
                "tracking_only",
                "research_only",
            } or obj.get("betting_mode") in {
                "paper_trading",
                "paper_tracking",
                "shadow_tracking",
            }, f"stake_multiplier appears without paper-only context in {path}"
