# Evaluation

## Why evaluation matters

The goal is not merely to predict winners. A sports prediction system must demonstrate that it can be compared against the market, tracked over time, and evaluated without leakage.

## Brier score

Brier score measures the squared error between predicted probability and outcome.

Lower is better.

```text
Brier = mean((predicted_probability - outcome)^2)
