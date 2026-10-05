"""
Evaluation metrics for regression predictions.

MAE      average size of a miss, in the stat's own units ("off by 4.6 points")
RMSE     like MAE but punishes big misses more
R2       share of the game-to-game variation the model explains (1.0 = perfect)
BIAS     average (prediction - actual); > 0 means the model guesses too high
COVERAGE share of actual results that landed inside the predicted range
"""

import numpy as np


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    total_variation = np.sum((actual - actual.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": float(1 - np.sum(errors ** 2) / total_variation),
        "bias": float(np.mean(errors)),
        "n": int(len(actual)),
    }


def interval_metrics(actual, lower, upper):
    actual = np.asarray(actual, dtype=float)
    inside = (actual >= lower) & (actual <= upper)
    return {
        "coverage": float(np.mean(inside)),
        "avg_width": float(np.mean(np.asarray(upper) - np.asarray(lower))),
    }
