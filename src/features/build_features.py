"""
Feature engineering: turn game logs into model-ready rows.

THE RULE: every feature on a row is computed ONLY from games played before
that row's game. The prediction for game N never sees game N's result.
That is what `shift(1)` does below, and tests/test_features.py proves it.

Feature groups (used later for the ablation study):
    form          the player's own recent production (rolling averages, trends)
    context       home/away, rest, back-to-back, place in the season
    opponent      how much the opponent has been allowing lately, and its pace
    team          the player's own team's offence and pace
    availability  minutes missing from the lineup (a stand-in for injury reports)

Run from the project root:
    python -m src.features.build_features            # read from PostgreSQL
    python -m src.features.build_features --source csv
"""

import argparse

import numpy as np
import pandas as pd

from src.config import FEATURES_PATH, MIN_PRIOR_GAMES, TARGETS

# Player statistics we compute rolling averages for.
ROLL_STATS = ["PTS", "REB", "AST", "FG3M", "MIN", "FGA", "FG3A", "FTA", "OREB", "TOV"]
WINDOWS = [3, 5, 10]

# Team-level statistics (summed over a team's players in one game).
TEAM_STATS = ["PTS", "REB", "AST", "FG3M", "FGA", "FTA", "OREB", "TOV"]
TEAM_WINDOW = 10

# Rest days are capped: 4 days and 150 days (off-season) both mean "rested".
REST_CAP = 7

# How many of a team's most recent games a player must have appeared in to
# count as "missing" when he does not play.
RECENT_TEAM_GAMES = 3


# ---------------------------------------------------------------------------
# 0. Make both data sources look the same
# ---------------------------------------------------------------------------

def standardize(df):
    """
    Accept game logs from the CSVs or from PostgreSQL and return one format.
    The CSV path lacks NEUTRAL_SITE and OPPONENT_TEAM_ID, so derive them.
    """
    df = df.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    # The database stores minutes with 2 decimals; match that everywhere so
    # both sources give identical features.
    df["MIN"] = df["MIN"].astype(float).round(2)

    if "OPPONENT_TEAM_ID" not in df.columns:
        abbr_to_id = dict(zip(df["TEAM_ABBREVIATION"], df["TEAM_ID"]))
        df["OPPONENT_TEAM_ID"] = df["OPPONENT"].map(abbr_to_id)
    if "NEUTRAL_SITE" not in df.columns:
        # Neutral site = neither team in the game is marked as home.
        home_teams_in_game = df.groupby("GAME_ID")["HOME"].transform("max")
        df["NEUTRAL_SITE"] = (home_teams_in_game == 0).astype(int)

    keep = ["SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
            "GAME_ID", "GAME_DATE", "HOME", "NEUTRAL_SITE", "OPPONENT_TEAM_ID",
            "OPPONENT", "STL", "BLK", *ROLL_STATS]
    df = df[keep]
    # Stable chronological order. GAME_ID breaks ties on the same date.
    return df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Player form
# ---------------------------------------------------------------------------

def add_player_form_features(df):
    """Rolling averages, season averages, volatility, trends and rates."""
    by_player = df.groupby("PLAYER_ID")

    for stat in ROLL_STATS:
        # shift(1): the row for game N only "sees" games 1..N-1.
        previous = by_player[stat].shift(1)
        for window in WINDOWS:
            df[f"{stat}_L{window}"] = (
                previous.groupby(df["PLAYER_ID"])
                        .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )

    # Season-to-date average (resets each season). NaN in a player's first
    # game of a season, because nothing has happened yet that season.
    by_player_season = df.groupby(["PLAYER_ID", "SEASON_YEAR"])
    for stat in TARGETS:
        previous = by_player_season[stat].shift(1)
        df[f"{stat}_SEASON_AVG"] = (
            previous.groupby([df["PLAYER_ID"], df["SEASON_YEAR"]])
                    .transform(lambda x: x.expanding().mean())
        )
        # Volatility: how much this stat has bounced around over 10 games.
        df[f"{stat}_STD_L10"] = (
            by_player[stat].shift(1).groupby(df["PLAYER_ID"])
                           .transform(lambda x: x.rolling(10, min_periods=3).std())
        )

    # Trend = short-term form minus longer-term form (your original feature).
    df["PTS_TREND"] = df["PTS_L3"] - df["PTS_L10"]
    df["MIN_TREND"] = df["MIN_L3"] - df["MIN_L10"]

    # Per-minute rates separate "plays a lot" from "produces a lot".
    minutes = df["MIN_L10"].replace(0, np.nan)
    df["PTS_PER_MIN_L10"] = df["PTS_L10"] / minutes
    df["REB_PER_MIN_L10"] = df["REB_L10"] / minutes
    df["AST_PER_MIN_L10"] = df["AST_L10"] / minutes
    # Rough usage: shots + free-throw trips + turnovers per minute.
    df["USAGE_PER_MIN_L10"] = (df["FGA_L10"] + 0.44 * df["FTA_L10"] + df["TOV_L10"]) / minutes

    # How much history stands behind these numbers.
    df["GAMES_PLAYED_PRIOR"] = by_player.cumcount()
    return df


# ---------------------------------------------------------------------------
# 2. Game context
# ---------------------------------------------------------------------------

def add_context_features(df):
    by_player = df.groupby("PLAYER_ID")
    df["SEASON_GAME_NUMBER"] = df.groupby(["PLAYER_ID", "SEASON_YEAR"]).cumcount() + 1

    days = by_player["GAME_DATE"].diff().dt.days
    df["FIRST_GAME_OF_SEASON"] = (df["SEASON_GAME_NUMBER"] == 1).astype(int)
    # Off-season gaps and first-ever games are "fully rested".
    df["REST_DAYS"] = days.fillna(REST_CAP).clip(upper=REST_CAP)
    df.loc[df["FIRST_GAME_OF_SEASON"] == 1, "REST_DAYS"] = REST_CAP
    df["BACK_TO_BACK"] = (df["REST_DAYS"] == 1).astype(int)
    return df


# ---------------------------------------------------------------------------
# 3 & 4. Team and opponent strength (from team-game totals)
# ---------------------------------------------------------------------------

def build_team_game_table(df):
    """
    One row per team per game with leakage-safe rolling team numbers:
      TEAM_*      what the team has been producing (offence)
      ALLOWED_*   what the team has been giving up (defence)
      *_POSS_*    possessions per game, i.e. pace
    Every rolling value uses only the team's EARLIER games (shift(1)).
    """
    totals = (df.groupby(["GAME_ID", "GAME_DATE", "TEAM_ID", "OPPONENT_TEAM_ID"], as_index=False)
                [TEAM_STATS].sum())
    totals["POSS"] = totals["FGA"] + 0.44 * totals["FTA"] - totals["OREB"] + totals["TOV"]

    # What each team ALLOWED in a game = what its opponent produced in it.
    opponent_side = totals[["GAME_ID", "TEAM_ID", *TEAM_STATS, "POSS"]].rename(
        columns={"TEAM_ID": "OPPONENT_TEAM_ID",
                 **{s: f"ALLOWED_{s}" for s in [*TEAM_STATS, "POSS"]}})
    team_games = totals.merge(opponent_side, on=["GAME_ID", "OPPONENT_TEAM_ID"])
    team_games = team_games.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    rolled_columns = {
        "PTS": "TEAM_PTS_L10", "POSS": "TEAM_POSS_L10",
        "ALLOWED_PTS": "ALLOWED_PTS_L10", "ALLOWED_REB": "ALLOWED_REB_L10",
        "ALLOWED_AST": "ALLOWED_AST_L10", "ALLOWED_FG3M": "ALLOWED_FG3M_L10",
        "ALLOWED_POSS": "ALLOWED_POSS_L10",
    }
    by_team = team_games.groupby("TEAM_ID")
    for source, name in rolled_columns.items():
        team_games[name] = (by_team[source].shift(1).groupby(team_games["TEAM_ID"])
                            .transform(lambda x: x.rolling(TEAM_WINDOW, min_periods=1).mean()))
    return team_games[["GAME_ID", "TEAM_ID", *rolled_columns.values()]]


def add_team_and_opponent_features(df):
    team_games = build_team_game_table(df)

    # The player's own team: offence and pace.
    own = team_games[["GAME_ID", "TEAM_ID", "TEAM_PTS_L10", "TEAM_POSS_L10"]]
    df = df.merge(own, on=["GAME_ID", "TEAM_ID"], how="left")

    # The opponent: what it allows, and its pace.
    opp_cols = ["ALLOWED_PTS_L10", "ALLOWED_REB_L10", "ALLOWED_AST_L10",
                "ALLOWED_FG3M_L10", "ALLOWED_POSS_L10"]
    opp = team_games[["GAME_ID", "TEAM_ID", *opp_cols, "TEAM_POSS_L10"]].rename(
        columns={"TEAM_ID": "OPPONENT_TEAM_ID", "TEAM_POSS_L10": "OPP_POSS_L10",
                 **{c: f"OPP_{c}" for c in opp_cols}})
    df = df.merge(opp, on=["GAME_ID", "OPPONENT_TEAM_ID"], how="left")
    return df


# ---------------------------------------------------------------------------
# 5. Lineup availability (a stand-in for injury reports)
# ---------------------------------------------------------------------------

def build_missing_minutes(df):
    """
    For every team-game: the recent minutes-per-game of rotation players who
    are NOT playing tonight.

    A player counts as missing if he played for this team in one of its last
    3 games, is still on the team (his latest game was for this team), and
    does not appear in tonight's box score. His weight is his 10-game
    minutes average, so missing a star counts far more than missing a
    bench player.

    Pre-game availability: the NBA publishes each team's inactive list
    before tip-off, so "who is out tonight" is known when predicting.
    Only earlier games feed the minutes weights.
    """
    # Minutes average INCLUDING the game just played = what is known going
    # into the player's next game.
    df = df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"])
    min_after = (df.groupby("PLAYER_ID")["MIN"]
                   .transform(lambda x: x.rolling(10, min_periods=1).mean()))
    player_rows = df.assign(MIN_AFTER=min_after.values)

    games_order = (player_rows[["GAME_DATE", "GAME_ID", "TEAM_ID", "SEASON_YEAR"]]
                   .drop_duplicates()
                   .sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]))
    lineups = {key: grp for key, grp in
               player_rows.groupby(["GAME_ID", "TEAM_ID"])[["PLAYER_ID", "MIN_AFTER"]]}

    team_game_count = {}   # (team, season) -> games played so far this season
    last_seen = {}         # player -> (team, season, team's game index, minutes avg)
    results = []

    for date, day in games_order.groupby("GAME_DATE", sort=True):
        teams_today = list(zip(day["GAME_ID"], day["TEAM_ID"], day["SEASON_YEAR"]))
        # 1) Measure absences for every game on this date, using only state
        #    built from EARLIER dates. Only players seen with this team THIS
        #    season count, so off-season departures are never "missing".
        for game_id, team_id, season in teams_today:
            index = team_game_count.get((team_id, season), 0)
            playing = set(lineups[(game_id, team_id)]["PLAYER_ID"])
            missing = sum(
                minutes
                for player, (team, seen_season, seen_index, minutes) in last_seen.items()
                if team == team_id and seen_season == season and player not in playing
                and seen_index >= index - RECENT_TEAM_GAMES
            )
            results.append((game_id, team_id, missing))
        # 2) Then update the state with tonight's games.
        for game_id, team_id, season in teams_today:
            index = team_game_count.get((team_id, season), 0)
            lineup = lineups[(game_id, team_id)]
            for player, minutes in zip(lineup["PLAYER_ID"], lineup["MIN_AFTER"]):
                last_seen[player] = (team_id, season, index, minutes)
            team_game_count[(team_id, season)] = index + 1

    return pd.DataFrame(results, columns=["GAME_ID", "TEAM_ID", "TEAM_MISSING_MIN"])


def add_availability_features(df):
    missing = build_missing_minutes(df)
    df = df.merge(missing, on=["GAME_ID", "TEAM_ID"], how="left")
    opp_missing = missing.rename(columns={"TEAM_ID": "OPPONENT_TEAM_ID",
                                          "TEAM_MISSING_MIN": "OPP_MISSING_MIN"})
    return df.merge(opp_missing, on=["GAME_ID", "OPPONENT_TEAM_ID"], how="left")


# ---------------------------------------------------------------------------
# Feature lists and the full pipeline
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "form": (
        [f"{s}_L{w}" for s in ROLL_STATS for w in WINDOWS]
        + [f"{t}_SEASON_AVG" for t in TARGETS]
        + [f"{t}_STD_L10" for t in TARGETS]
        + ["PTS_TREND", "MIN_TREND", "PTS_PER_MIN_L10", "REB_PER_MIN_L10",
           "AST_PER_MIN_L10", "USAGE_PER_MIN_L10", "GAMES_PLAYED_PRIOR"]
    ),
    "context": ["HOME", "NEUTRAL_SITE", "REST_DAYS", "BACK_TO_BACK",
                "FIRST_GAME_OF_SEASON", "SEASON_GAME_NUMBER"],
    "opponent": ["OPP_ALLOWED_PTS_L10", "OPP_ALLOWED_REB_L10", "OPP_ALLOWED_AST_L10",
                 "OPP_ALLOWED_FG3M_L10", "OPP_ALLOWED_POSS_L10", "OPP_POSS_L10"],
    "team": ["TEAM_PTS_L10", "TEAM_POSS_L10"],
    "availability": ["TEAM_MISSING_MIN", "OPP_MISSING_MIN"],
}
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


def build_features(game_logs):
    """Game logs in, one feature row per player-game out (all rows kept)."""
    df = standardize(game_logs)
    df = add_player_form_features(df)
    df = add_context_features(df)
    df = add_team_and_opponent_features(df)
    df = add_availability_features(df)
    return df.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)


def model_rows(features):
    """Rows usable for training/testing: the player has enough history."""
    return features[features["GAMES_PLAYED_PRIOR"] >= MIN_PRIOR_GAMES].copy()


def main():
    parser = argparse.ArgumentParser(description="Build the feature dataset.")
    parser.add_argument("--source", choices=["db", "csv"], default="db",
                        help="Read game logs from PostgreSQL (default) or the processed CSVs.")
    args = parser.parse_args()

    if args.source == "db":
        from src.database.queries import read_game_logs_from_db
        game_logs = read_game_logs_from_db()
    else:
        from src.ingestion.nba_stats import load_processed
        game_logs = load_processed()

    features = build_features(game_logs)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Built {len(ALL_FEATURES)} features for {len(features):,} player-games "
          f"({len(model_rows(features)):,} with >= {MIN_PRIOR_GAMES} prior games).")
    print(f"Saved: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
