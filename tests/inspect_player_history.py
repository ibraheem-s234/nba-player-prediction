import pandas as pd

DATA_PATH = "data/processed/player_game_logs_2024_25.csv"


def add_rolling_features(df, column, windows):
    """
    Add leakage-safe rolling averages for a statistic.

    Each feature uses only games that happened before
    the current game.
    """

    for window in windows:
        feature_name = f"{column}_L{window}"

        df[feature_name] = (
            df.groupby("PLAYER_ID")[column]
              .transform(
                  lambda x: x.shift(1).rolling(window).mean()
              )
        )

    return df


df = pd.read_csv(DATA_PATH)

# Convert dates to datetime.
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

# Sort each player's games chronologically.
df = df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"])

# Create rolling point features.
df = add_rolling_features(
    df,
    "PTS",
    [3, 5, 10]
)

df = add_rolling_features(
    df,
    "MIN",
    [3, 5, 10]
)

df = add_rolling_features(
    df,
    "REB",
    [3, 5, 10]
)

df = add_rolling_features(
    df,
    "AST",
    [3, 5, 10]
)

df["REST_DAYS"] = (
    df.groupby("PLAYER_ID")["GAME_DATE"]
      .diff()
      .dt.days
)

df["PTS_TREND"] = df["PTS_L3"] - df["PTS_L10"]

# Inspect Stephen Curry.
curry = df[df["PLAYER_NAME"] == "Stephen Curry"]

print(
    curry[
        [
            "GAME_DATE",
            "PTS",
            "MIN",
            "REB",
            "AST",
            "PTS_L3",
            "PTS_L5",
            "PTS_L10",
            "PTS_TREND",
            "MIN_L3",
            "MIN_L5",
            "MIN_L10",
            "REB_L3",
            "REB_L5",
            "REB_L10",
            "AST_L3",
            "AST_L5",
            "AST_L10",
            "REST_DAYS",
        ]
    ]
    .tail(15)
    .to_string(index=False)
)

print("\nOpponent frequency:")
print(
    df["OPPONENT"]
    .value_counts()
    .sort_index()
)