"""
Next-game projections + data export for the dashboard.

How a projection is made (the same feature code as training, so no shortcuts):
  1. Take every active player's real game history.
  2. Append ONE upcoming game for each team, with empty stat lines.
  3. Run the normal feature pipeline. Because every feature uses shift(1),
     the upcoming row only "sees" real, already-played games.
  4. Swap in each possible opponent (and home/away), then ask the saved
     models for a median projection and an 80% range.

The scenario ("the next game, as if the latest season continued"):
  - the player's end-of-season form, season averages and team carry over;
  - the game is two days after the season's final game (normal rest);
  - both teams have a typical night of absences: the median "minutes
    missing" seen across all training games.
Why not "the first game of next season"? Only ~1,800 of 126,000 training
rows are season openers, and they have no season average yet, so the model
would be asked about a rare situation it has little evidence for.

Run from the project root (after training):
    python -m src.predictions.predict
"""

import json
from datetime import date

import joblib
import pandas as pd

from src.config import (MODELS_DIR, NEXT_GAME_DAYS_AFTER_SEASON, PROJECTION_MIN_GAMES,
                        PROJECTION_MIN_MINUTES, REPORTS_DIR, SEASONS, SITE_DATA_DIR,
                        TARGETS, TEST_SEASON)
from src.features.build_features import ALL_FEATURES, build_features, standardize
from src.ingestion.nba_stats import load_processed
from src.models import intervals

OPPONENT_COLUMNS = ["OPP_ALLOWED_PTS_L10", "OPP_ALLOWED_REB_L10", "OPP_ALLOWED_AST_L10",
                    "OPP_ALLOWED_FG3M_L10", "OPP_ALLOWED_POSS_L10", "OPP_POSS_L10"]


def active_rosters(logs):
    """Players worth projecting, each assigned to the team of his latest game."""
    latest = logs[logs["SEASON_YEAR"] == SEASONS[-1]].sort_values(["GAME_DATE", "GAME_ID"])
    games = latest.groupby("PLAYER_ID").size()
    recent_minutes = latest.groupby("PLAYER_ID")["MIN"].apply(lambda m: m.tail(10).mean())
    last_row = latest.groupby("PLAYER_ID").tail(1).set_index("PLAYER_ID")

    keep = games[(games >= PROJECTION_MIN_GAMES)].index.intersection(
        recent_minutes[recent_minutes >= PROJECTION_MIN_MINUTES].index)
    return last_row.loc[keep, ["PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]].reset_index()


def upcoming_rows(rosters, teams, game_date):
    """
    One placeholder game per pair of teams, all on game_date, with empty
    stat lines. Opponents are re-assigned later, so the pairing is arbitrary.
    """
    team_ids = sorted(rosters["TEAM_ID"].unique())
    pairs = [(team_ids[i], team_ids[i + 1]) for i in range(0, len(team_ids) - 1, 2)]
    rows = []
    for n, (home, away) in enumerate(pairs):
        game_id = f"NEXT{n:06d}"
        for team, opp, is_home in [(home, away, 1), (away, home, 0)]:
            for player in rosters[rosters["TEAM_ID"] == team].itertuples():
                rows.append({
                    "SEASON_YEAR": SEASONS[-1], "PLAYER_ID": player.PLAYER_ID,
                    "PLAYER_NAME": player.PLAYER_NAME, "TEAM_ID": team,
                    "TEAM_ABBREVIATION": teams[team], "GAME_ID": game_id,
                    "GAME_DATE": game_date, "HOME": is_home,
                    "OPPONENT": teams[opp], "OPPONENT_TEAM_ID": opp, "NEUTRAL_SITE": 0,
                })
    return pd.DataFrame(rows)


def project(logs):
    """Return (player feature rows, team defence table) for the upcoming game."""
    teams = dict(zip(logs["TEAM_ID"], logs["TEAM_ABBREVIATION"]))
    rosters = active_rosters(logs)

    history = standardize(logs)
    game_date = history["GAME_DATE"].max() + pd.Timedelta(days=NEXT_GAME_DAYS_AFTER_SEASON)
    upcoming = upcoming_rows(rosters, teams, game_date)
    combined = pd.concat([history, upcoming], ignore_index=True)
    features = build_features(combined)
    next_rows = features[features["GAME_ID"].str.startswith("NEXT")].copy()

    # Each team's defensive profile going into its next game, read from the
    # rows where it appears as the opponent.
    defence = (next_rows.groupby("OPPONENT_TEAM_ID")[OPPONENT_COLUMNS].first())
    return next_rows, defence, teams


def predict_matchups(next_rows, defence, models, missing_minutes):
    """Predict every player against every opponent, at home and away."""
    base = next_rows.drop(columns=OPPONENT_COLUMNS + ["OPPONENT_TEAM_ID"])
    base["TEAM_MISSING_MIN"] = missing_minutes   # a typical night of absences
    base["OPP_MISSING_MIN"] = missing_minutes
    base["NEUTRAL_SITE"] = 0

    grid = []
    for opponent_id, opp_values in defence.iterrows():
        for home in (1, 0):
            rows = base[base["TEAM_ID"] != opponent_id].copy()
            rows["OPPONENT_TEAM_ID"] = opponent_id
            rows["HOME"] = home
            for column in OPPONENT_COLUMNS:
                rows[column] = opp_values[column]
            grid.append(rows)
    grid = pd.concat(grid, ignore_index=True)

    X = grid[ALL_FEATURES]
    for target, saved in models.items():
        predicted = saved["model"].predict(X)
        low, high = intervals.predict_range(grid, predicted, target, saved["range"])
        grid[f"{target}_PRED"] = predicted
        grid[f"{target}_LOW"] = low
        grid[f"{target}_HIGH"] = high
    return grid


# ---------------------------------------------------------------------------
# Export for the dashboard
# ---------------------------------------------------------------------------

def r1(x):
    """Round to one decimal (keeps the JSON small) and never go below zero."""
    return round(max(float(x), 0.0), 1)


def compact(x, whole=False):
    """Box-score counts are whole numbers: write 24, not 24.0."""
    return int(round(float(x))) if whole else r1(x)


def export_site_data(logs, grid, teams, missing_minutes, game_date):
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    targets = list(TARGETS)
    full_names = dict(zip(logs["TEAM_ID"], logs["TEAM_NAME"]))
    latest = logs[logs["SEASON_YEAR"] == SEASONS[-1]]

    backtest = pd.read_csv(REPORTS_DIR / "backtest_predictions.csv")
    backtest_by_player = dict(tuple(backtest.groupby("PLAYER_ID")))

    players = []
    for player_id, rows in grid.groupby("PLAYER_ID"):
        first = rows.iloc[0]
        season_rows = latest[latest["PLAYER_ID"] == player_id].sort_values("GAME_DATE")

        projections = {}
        for r in rows.itertuples():
            side = "home" if r.HOME == 1 else "away"
            projections.setdefault(teams[r.OPPONENT_TEAM_ID], {})[side] = [
                [r1(getattr(r, f"{t}_PRED")), r1(getattr(r, f"{t}_LOW")),
                 r1(getattr(r, f"{t}_HIGH"))] for t in targets]

        last10 = season_rows.tail(10)
        player = {
            "id": int(player_id),
            "name": first["PLAYER_NAME"],
            "team": first["TEAM_ABBREVIATION"],
            "gp": int(len(season_rows)),
            "avg": [r1(season_rows[t].mean()) for t in targets],
            "last10": {
                "date": last10["GAME_DATE"].dt.strftime("%Y-%m-%d").tolist(),
                "opp": last10["OPPONENT"].tolist(),
                **{t: [r1(v) for v in last10[t]] for t in targets},
            },
            "proj": projections,
        }

        bt = backtest_by_player.get(player_id)
        if bt is not None:
            bt = bt.sort_values("GAME_DATE")
            player["backtest"] = {
                "date": bt["GAME_DATE"].tolist(),
                "opp": bt["OPPONENT"].tolist(),
                "home": bt["HOME"].astype(int).tolist(),
                **{f"{t}_{kind}": [compact(v, whole=(kind == "actual" and t != "MIN"))
                                    for v in bt[f"{t}_{kind.upper()}"]]
                   for t in targets for kind in ("actual", "pred", "low", "high")},
            }
        players.append(player)

    # Roster order on the dashboard: projected points against an average opponent.
    def average_points(p):
        return sum(v["home"][0][0] for v in p["proj"].values()) / len(p["proj"])
    players.sort(key=average_points, reverse=True)

    payload = {
        "generated": date.today().isoformat(),
        "data_seasons": SEASONS,
        "n_player_games": int(len(logs)),
        "n_games": int(logs["GAME_ID"].nunique()),
        "test_season": TEST_SEASON,
        "scenario": {"missing_minutes": round(missing_minutes, 1),
                     "game_date": game_date},
        "targets": targets,
        "target_labels": [TARGETS[t] for t in targets],
        "teams": {abbr: full_names[tid] for tid, abbr in teams.items()},
        "players": players,
    }
    (SITE_DATA_DIR / "players.json").write_text(json.dumps(payload, separators=(",", ":")))

    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())
    (SITE_DATA_DIR / "metrics.json").write_text(json.dumps(metrics, separators=(",", ":")))
    print(f"Exported {len(players)} players to {SITE_DATA_DIR}")


def main():
    logs = load_processed()
    models = {t: joblib.load(MODELS_DIR / f"{t.lower()}_model.joblib") for t in TARGETS}
    for t, saved in models.items():
        if saved["features"] != ALL_FEATURES:
            raise RuntimeError(f"Saved {t} model was trained on different features; retrain it.")

    next_rows, defence, teams = project(logs)
    # Median minutes missing per team-game in the training data (saved at training time).
    missing_minutes = models["PTS"]["typical_missing_min"]
    grid = predict_matchups(next_rows, defence, models, missing_minutes)
    game_date = next_rows["GAME_DATE"].iloc[0].strftime("%Y-%m-%d")
    export_site_data(logs, grid, teams, missing_minutes, game_date)


if __name__ == "__main__":
    main()
