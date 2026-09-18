

from pathlib import Path

import pandas as pd

from nba_api.stats.endpoints import playergamelogs


def fetch_player_game_logs(season: str = "2024-25"):
    """Fetch regular-season NBA player game logs for one season."""

    print(f"Fetching NBA player game logs for {season}...")

    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable="Regular Season",
    )

    return logs.get_data_frames()[0]


def main():
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = fetch_player_game_logs()

    columns = [
    "SEASON_YEAR",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "PF",
    "PFD",
    "PTS",
    "PLUS_MINUS",
]

    df = df[columns]

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df["HOME"] = df["MATCHUP"].str.contains("vs.").astype(int)
    df["OPPONENT"] = df["MATCHUP"].str[-3:] 

    print("\nMissing values:")
    print(df.isna().sum()[df.isna().sum() > 0])

    print("\nDuplicate player-game rows:")
    print(df.duplicated(subset=["PLAYER_ID", "GAME_ID"]).sum())


    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_file = processed_dir / "player_game_logs_2024_25.csv"
    df.to_csv(output_file, index=False)

    print(f"\nDownloaded {len(df)} player-game rows.")
    print(f"Saved to: {output_file}")
    print("\nFirst five rows:")
    print(df.head())


if __name__ == "__main__":
    main()