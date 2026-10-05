import pandas as pd

df = pd.read_csv("data/processed/player_game_logs_2024_25.csv")

problem_games = [
    "22400147",
    "22400621",
    "22400633",
    "22401229",
    "22401230"
]

for game_id in problem_games:
    print(f"\n{'=' * 60}")
    print(f"GAME: {game_id}")
    print(f"{'=' * 60}")

    game = df[df["GAME_ID"].astype(str) == game_id]

    print(
        game[
            [
                "GAME_ID",
                "GAME_DATE",
                "TEAM_ID",
                "TEAM_ABBREVIATION",
                "MATCHUP",
                "HOME",
                "OPPONENT"
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )