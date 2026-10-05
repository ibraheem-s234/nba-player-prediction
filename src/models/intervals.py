"""
Prediction ranges ("80% of the time the real result lands in here").

Method: split conformal prediction, scaled by each player's volatility.

  1. A model trained WITHOUT the validation season predicts it.
  2. For each validation game: error = (actual - predicted) / volatility,
     where volatility = the player's standard deviation over his last 10
     games. Dividing by it puts steady and streaky players on one scale.
  3. Find the band of scaled errors [lo, hi] that contains 80% of the
     validation results.
  4. For a new prediction:  range = prediction + [lo, hi] x volatility.

So a streaky scorer gets a wide range and a steady one a narrow range, and
the 80% is checked on data the model never trained on.
"""

import numpy as np

from src.models.evaluate import interval_metrics

TARGET_COVERAGE = 0.80
# Candidate tail sizes. Stats are whole numbers, so a band that leaves out
# exactly 10% on each side can still contain more than 80% (ties at the
# edges count as inside); we try a few and keep the closest to 80%.
TAIL_SIZES = [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]


def volatility(rows, target, fill, floor):
    """The player's recent standard deviation, with sensible defaults."""
    return rows[f"{target}_STD_L10"].fillna(fill).clip(lower=floor).to_numpy()


def calibrate(train, valid, valid_predictions, target):
    """Learn the band from validation errors. Returns the settings to store."""
    stds = train[f"{target}_STD_L10"].dropna()
    settings = {"fill": float(stds.median()), "floor": float(stds.quantile(0.10))}

    scale = volatility(valid, target, settings["fill"], settings["floor"])
    scaled_errors = (valid[target].to_numpy() - valid_predictions) / scale

    best = None
    for tail in TAIL_SIZES:
        lo, hi = np.quantile(scaled_errors, [tail, 1 - tail])
        low, high = apply(valid_predictions, scale, lo, hi)
        coverage = interval_metrics(valid[target], low, high)["coverage"]
        if best is None or abs(coverage - TARGET_COVERAGE) < abs(best["coverage"] - TARGET_COVERAGE):
            best = {"tail": tail, "lo": float(lo), "hi": float(hi), "coverage": coverage}

    settings.update({"lo": best["lo"], "hi": best["hi"], "tail": best["tail"],
                     "validation_coverage": best["coverage"]})
    return settings


def apply(predictions, scale, lo, hi):
    """Turn point predictions into (low, high) ranges. Stats can't go below 0."""
    low = np.maximum(predictions + lo * scale, 0)
    high = np.maximum(predictions + hi * scale, low)
    return low, high


def predict_range(rows, predictions, target, settings):
    scale = volatility(rows, target, settings["fill"], settings["floor"])
    return apply(np.asarray(predictions), scale, settings["lo"], settings["hi"])
