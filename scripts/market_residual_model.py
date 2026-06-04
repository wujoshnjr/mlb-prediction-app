from __future__ import annotations

import math
from typing import Any, Optional

import lightgbm as lgb
import numpy as np


def compute_no_vig_probability(home_odds: float, away_odds: float) -> float:
    try:
        home = float(home_odds)
        away = float(away_odds)
        if home <= 1 or away <= 1:
            return 0.5
        p_home_raw = 1.0 / home
        p_away_raw = 1.0 / away
        total = p_home_raw + p_away_raw
        if total <= 0:
            return 0.5
        return float(p_home_raw / total)
    except Exception:
        return 0.5


def probability_to_logit(probability: float, epsilon: float = 1e-7) -> float:
    probability = float(np.clip(probability, epsilon, 1.0 - epsilon))
    return float(np.log(probability / (1.0 - probability)))


def logit_to_probability(logit_value: float) -> float:
    logit_value = float(np.clip(logit_value, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-logit_value)))


def train_residual_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    p_train_market_no_vig: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    p_val_market_no_vig: np.ndarray,
) -> lgb.Booster:
    train_margin = np.array(
        [probability_to_logit(float(prob)) for prob in p_train_market_no_vig],
        dtype=float,
    )
    val_margin = np.array(
        [probability_to_logit(float(prob)) for prob in p_val_market_no_vig],
        dtype=float,
    )

    train_data = lgb.Dataset(x_train, label=y_train, init_score=train_margin)
    val_data = lgb.Dataset(x_val, label=y_val, init_score=val_margin)

    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 4,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    return lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )


def predict_model_probability(
    model: lgb.Booster,
    features: np.ndarray,
    market_no_vig_probability: float,
) -> float:
    market_logit = probability_to_logit(market_no_vig_probability)
    delta_logit = float(model.predict(features, raw_score=True)[0])
    return logit_to_probability(market_logit + delta_logit)
