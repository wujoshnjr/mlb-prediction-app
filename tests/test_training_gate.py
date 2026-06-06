from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import MIN_CLEAN_TRAIN_SAMPLES


TRAINING_STATUS_PATH = Path("data/training_status.json")


def test_min_clean_train_samples_is_300() -> None:
    assert MIN_CLEAN_TRAIN_SAMPLES == 300


def test_training_status_does_not_train_below_300() -> None:
    if not TRAINING_STATUS_PATH.exists():
        pytest.skip("training_status.json not found")

    data = json.loads(TRAINING_STATUS_PATH.read_text(encoding="utf-8"))
    sample_count = int(data.get("sample_count") or 0)

    if sample_count < 300:
        assert data.get("trained") is False
        assert data.get("skipped") is True
        assert data.get("training_allowed_for_production") is False
