import pandas as pd

df = pd.read_csv("data/processed/player_game_logs_2024_25.csv")

print("Rows:", len(df))
print("\nMIN examples:")
print(df["MIN"].head(20).to_list())

print("\nMIN data type:", df["MIN"].dtype)
print("\nMissing MIN values:", df["MIN"].isna().sum())

print("\nSample statistics:")
print(
    df[["PLAYER_ID", "GAME_ID", "MIN", "PTS", "REB", "AST"]]
    .head(10)
    .to_string(index=False)
)