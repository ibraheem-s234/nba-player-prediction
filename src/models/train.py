"""
Train, compare, and backtest the prediction models.

Evaluation protocol (no information from the future is ever used):

    seasons:   2021-22  2022-23  2023-24 | 2024-25    | 2025-26
    step 1:    [------- train --------]  [validate]               choose settings,
                                                                  calibrate ranges
    step 2:    [--------------- train ---------------] [  TEST  ]  final, honest score
    step 3:    [------------------ train on everything ------------] models used by the app

For every target (PTS, REB, AST, FG3M, MIN) we compare:
    Season average     the player's average so far this season (the "naive" guess)
    Last 10 games      the player's average over his last 10 games
    Ridge              linear regression with a penalty against extreme weights
    Random forest      many decision trees, averaged
    Gradient boosting  trees built one after another, each fixing the last one's errors

Run from the project root (after building features):
    python -m src.models.train
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (FEATURES_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR,
                        TARGETS, TEST_SEASON, TRAIN_SEASONS, VALIDATION_SEASON)
from src.features.build_features import ALL_FEATURES, FEATURE_GROUPS, model_rows
from src.models import intervals
from src.models.evaluate import interval_metrics, regression_metrics

# Candidate settings for gradient boosting, compared on the validation season.
#   learning_rate   how big a correction each new tree may make
#   max_leaf_nodes  how detailed each tree may be
#   max_iter        how many trees are built
BOOSTING_GRID = [
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "max_iter": 200},
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "max_iter": 400},
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 200},
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 400},
]

# A "rotation player" plays real minutes; these are the predictions people care about.
ROTATION_MINUTES = 20


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def boosting_model(params):
    """
    Gradient boosting with absolute-error loss predicts the MEDIAN outcome,
    which is exactly what minimises MAE, our headline metric.

    early_stopping=False: scikit-learn would otherwise switch it on for
    datasets over 10,000 rows and quietly ignore max_iter, which would make
    the settings comparison meaningless.
    """
    return HistGradientBoostingRegressor(
        loss="absolute_error", min_samples_leaf=40, early_stopping=False,
        random_state=RANDOM_STATE, **params)


def ridge_model():
    # Linear models cannot handle missing values or very different scales,
    # so we fill gaps with the median and standardise every feature first.
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def forest_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=100, min_samples_leaf=25, max_features=0.4,
                              max_samples=0.5, n_jobs=-1, random_state=RANDOM_STATE))


def baseline_predictions(rows, target):
    """The two no-ML baselines every model must beat."""
    season_avg = rows[f"{target}_SEASON_AVG"].fillna(rows[f"{target}_L10"])
    return {"Season average": season_avg.to_numpy(), "Last 10 games": rows[f"{target}_L10"].to_numpy()}


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def choose_boosting_params(train, valid, target):
    """
    Step 1: pick the settings with the lowest validation MAE.
    Returns the settings and that model's validation predictions (reused to
    calibrate the prediction ranges).
    """
    best = None
    for params in BOOSTING_GRID:
        model = boosting_model(params).fit(train[ALL_FEATURES], train[target])
        predicted = model.predict(valid[ALL_FEATURES])
        mae = regression_metrics(valid[target], predicted)["mae"]
        print(f"    {params} -> validation MAE {mae:.3f}")
        if best is None or mae < best[1]:
            best = (params, mae, predicted)
    return best[0], best[2]


def compare_models(train, test, target, params):
    """Step 2: fit every model on train, score all of them on the test season."""
    predictions = baseline_predictions(test, target)
    fitted = {}
    for name, model in [("Ridge", ridge_model()),
                        ("Random forest", forest_model()),
                        ("Gradient boosting", boosting_model(params))]:
        fitted[name] = model.fit(train[ALL_FEATURES], train[target])
        predictions[name] = model.predict(test[ALL_FEATURES])

    rotation = (test["MIN_L10"] >= ROTATION_MINUTES).to_numpy()
    results = {}
    for name, predicted in predictions.items():
        results[name] = regression_metrics(test[target], predicted)
        results[name]["mae_rotation"] = regression_metrics(
            test[target].to_numpy()[rotation], predicted[rotation])["mae"]
    return results, predictions["Gradient boosting"], fitted["Gradient boosting"]


def ablation(train, test, target, params):
    """
    Which information actually helps? Add one feature group at a time and
    measure test MAE. If a group does not lower the error, it is not
    pulling its weight.
    """
    results, used = [], []
    for group, columns in FEATURE_GROUPS.items():
        used = used + columns
        model = boosting_model(params).fit(train[used], train[target])
        mae = regression_metrics(test[target], model.predict(test[used]))["mae"]
        results.append({"added_group": group, "n_features": len(used), "mae": mae})
        print(f"    + {group:<12} ({len(used):>2} features) -> test MAE {mae:.3f}")
    return results


def feature_importance(model, test, target, top=15):
    """
    Permutation importance: shuffle one feature's column and see how much
    worse the predictions get. A big increase in error = an important feature.
    """
    sample = test.sample(n=min(5000, len(test)), random_state=RANDOM_STATE)
    result = permutation_importance(model, sample[ALL_FEATURES], sample[target],
                                    scoring="neg_mean_absolute_error", n_repeats=3,
                                    random_state=RANDOM_STATE)
    order = np.argsort(result.importances_mean)[::-1][:top]
    return [{"feature": ALL_FEATURES[i], "mae_increase": float(result.importances_mean[i])}
            for i in order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start = time.time()
    features = pd.read_csv(FEATURES_PATH, dtype={"GAME_ID": str}, parse_dates=["GAME_DATE"])
    rows = model_rows(features)

    train = rows[rows["SEASON_YEAR"].isin(TRAIN_SEASONS)]
    valid = rows[rows["SEASON_YEAR"] == VALIDATION_SEASON]
    train_and_valid = rows[rows["SEASON_YEAR"].isin(TRAIN_SEASONS + [VALIDATION_SEASON])]
    test = rows[rows["SEASON_YEAR"] == TEST_SEASON]
    print(f"Rows -> train {len(train):,} | validation {len(valid):,} | "
          f"train+validation {len(train_and_valid):,} | test {len(test):,}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {"test_season": TEST_SEASON, "train_seasons": TRAIN_SEASONS + [VALIDATION_SEASON],
              "validation_season": VALIDATION_SEASON, "n_features": len(ALL_FEATURES),
              "n_test_rows": int(len(test)), "rotation_minutes": ROTATION_MINUTES,
              "feature_groups": {g: len(c) for g, c in FEATURE_GROUPS.items()},
              "targets": {}}
    backtest = test[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID",
                     "GAME_DATE", "OPPONENT", "HOME"]].copy()

    for target, label in TARGETS.items():
        print(f"\n=== {label} ({target}) ===")

        # Step 1: settings + range calibration, using validation only.
        params, valid_pred = choose_boosting_params(train, valid, target)
        range_settings = intervals.calibrate(train, valid, valid_pred, target)
        print(f"  chosen: {params} | range band calibrated to "
              f"{range_settings['validation_coverage']:.1%} on validation")

        # Step 2: the honest test-season score.
        comparison, test_pred, test_model = compare_models(train_and_valid, test, target, params)
        for name, m in comparison.items():
            print(f"  {name:<18} MAE {m['mae']:.3f}  RMSE {m['rmse']:.3f}  "
                  f"R2 {m['r2']:.3f}  (rotation MAE {m['mae_rotation']:.3f})")
        low, high = intervals.predict_range(test, test_pred, target, range_settings)
        coverage = interval_metrics(test[target], low, high)
        print(f"  80% range: {coverage['coverage']:.1%} of test results inside "
              f"(average width {coverage['avg_width']:.1f})")

        importance = feature_importance(test_model, test, target)
        print("  ablation:")
        ablation_results = ablation(train_and_valid, test, target, params)

        report["targets"][target] = {
            "label": label, "params": params, "models": comparison,
            "interval": {**coverage, "validation_coverage": range_settings["validation_coverage"]},
            "ablation": ablation_results, "importance": importance,
        }
        backtest[f"{target}_ACTUAL"] = test[target].to_numpy()
        backtest[f"{target}_PRED"] = test_pred
        backtest[f"{target}_LOW"] = low
        backtest[f"{target}_HIGH"] = high

        # Step 3: the model the dashboard uses, trained on every season.
        final_model = boosting_model(params).fit(rows[ALL_FEATURES], rows[target])
        joblib.dump({
            "model": final_model,
            "range": range_settings,
            "features": ALL_FEATURES,
            # A "typical night" of absences, used by the projection scenario.
            "typical_missing_min": float(rows["TEAM_MISSING_MIN"].median()),
        }, MODELS_DIR / f"{target.lower()}_model.joblib")

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    backtest.to_csv(REPORTS_DIR / "backtest_predictions.csv", index=False,
                    date_format="%Y-%m-%d", float_format="%.3f")
    print(f"\nSaved models to {MODELS_DIR} and reports to {REPORTS_DIR} "
          f"({time.time() - start:.0f}s).")


if __name__ == "__main__":
    main()
