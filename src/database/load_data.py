"""
ETL load step: processed CSVs -> PostgreSQL.

Each table has two functions:
    build_<table>(df)   pure pandas: turns game-log rows into database records
                        (a list of dicts). No database needed, so it is unit-tested.
    insert_<table>(...) sends those records to PostgreSQL in one batch.

Run from the project root (after `python -m src.database.schema`):
    python -m src.database.load_data
"""

import pandas as pd

from src.ingestion.nba_stats import load_processed

# CSV column -> database column for player_game_stats.
STAT_COLUMNS = {
    "MIN": "minutes",
    "FGM": "field_goals_made",
    "FGA": "field_goals_attempted",
    "FG_PCT": "field_goal_percentage",
    "FG3M": "three_pointers_made",
    "FG3A": "three_pointers_attempted",
    "FG3_PCT": "three_point_percentage",
    "FTM": "free_throws_made",
    "FTA": "free_throws_attempted",
    "FT_PCT": "free_throw_percentage",
    "OREB": "offensive_rebounds",
    "DREB": "defensive_rebounds",
    "REB": "rebounds",
    "AST": "assists",
    "STL": "steals",
    "BLK": "blocks",
    "TOV": "turnovers",
    "PF": "personal_fouls",
    "PTS": "points",
    "PLUS_MINUS": "plus_minus",
}
DECIMAL_COLUMNS = {"minutes", "field_goal_percentage",
                   "three_point_percentage", "free_throw_percentage"}


# ---------------------------------------------------------------------------
# Build: pure transformations (no database)
# ---------------------------------------------------------------------------

def build_teams(df):
    """One record per team. If a name ever changes, the latest one wins."""
    latest = (df.sort_values("GAME_DATE")
                .drop_duplicates("TEAM_ID", keep="last"))
    return [
        {"team_id": int(r.TEAM_ID), "team_name": r.TEAM_NAME,
         "abbreviation": r.TEAM_ABBREVIATION}
        for r in latest.itertuples()
    ]


def build_players(df):
    """
    One record per player, using the most recent spelling of the name.
    (The data has e.g. 'Jonas Valanciunas' in one season and
    'Jonas Valančiūnas' in another -- same PLAYER_ID, so same person.)
    """
    latest = (df.sort_values("GAME_DATE")
                .drop_duplicates("PLAYER_ID", keep="last"))
    return [
        {"player_id": int(r.PLAYER_ID), "player_name": r.PLAYER_NAME}
        for r in latest.itertuples()
    ]


def build_games(df):
    """
    One record per game. The CSV has one row per PLAYER, so we first reduce
    to one row per TEAM per game, then combine the two teams.
    """
    team_rows = df[["GAME_ID", "SEASON_YEAR", "GAME_DATE", "TEAM_ID", "HOME"]].drop_duplicates()

    games = []
    for game_id, group in team_rows.groupby("GAME_ID"):
        if group["TEAM_ID"].nunique() != 2:
            raise ValueError(f"Game {game_id} does not have exactly two teams.")
        if group["GAME_DATE"].nunique() != 1:
            raise ValueError(f"Game {game_id} has multiple dates.")

        home = group.loc[group["HOME"] == 1, "TEAM_ID"].tolist()
        away = group.loc[group["HOME"] == 0, "TEAM_ID"].tolist()

        if len(home) == 1 and len(away) == 1:          # normal game
            home_id, away_id, neutral = int(home[0]), int(away[0]), False
        elif len(home) == 0 and len(away) == 2:        # neutral site (e.g. Paris, Mexico City)
            home_id, away_id, neutral = None, None, True
        else:
            raise ValueError(f"Game {game_id} has an unexpected home/away structure.")

        games.append({
            "game_id": str(game_id),
            "season": group["SEASON_YEAR"].iloc[0],
            "game_date": pd.Timestamp(group["GAME_DATE"].iloc[0]).date(),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "neutral_site": neutral,
        })
    return games


def build_player_game_stats(df):
    """One record per player per game, with plain Python types for the driver."""
    stats = df[["PLAYER_ID", "GAME_ID", "TEAM_ID", *STAT_COLUMNS]].rename(columns=STAT_COLUMNS)

    records = []
    for row in stats.to_dict("records"):
        record = {
            "player_id": int(row["PLAYER_ID"]),
            "game_id": str(row["GAME_ID"]),
            "team_id": int(row["TEAM_ID"]),
        }
        for column in STAT_COLUMNS.values():
            value = row[column]
            if pd.isna(value):
                record[column] = None
            elif column in DECIMAL_COLUMNS:
                record[column] = float(value)
            else:
                record[column] = int(value)
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Insert: write records to PostgreSQL
# ---------------------------------------------------------------------------

def _insert(sql, records):
    """
    Run one parameterised INSERT for every record, inside one transaction.
    Passing a list of dicts makes SQLAlchemy send them as a batch
    ("executemany"), which is much faster than one call per row.
    """
    from sqlalchemy import text
    from src.database.connection import engine

    with engine.begin() as connection:
        connection.execute(text(sql), records)


def insert_teams(records):
    _insert("""
        INSERT INTO teams (team_id, team_name, abbreviation)
        VALUES (:team_id, :team_name, :abbreviation)
        ON CONFLICT (team_id) DO UPDATE
            SET team_name = EXCLUDED.team_name,
                abbreviation = EXCLUDED.abbreviation;
    """, records)


def insert_players(records):
    _insert("""
        INSERT INTO players (player_id, player_name)
        VALUES (:player_id, :player_name)
        ON CONFLICT (player_id) DO UPDATE
            SET player_name = EXCLUDED.player_name;
    """, records)


def insert_games(records):
    _insert("""
        INSERT INTO games (game_id, season, game_date, home_team_id,
                           away_team_id, neutral_site)
        VALUES (:game_id, :season, :game_date, :home_team_id,
                :away_team_id, :neutral_site)
        ON CONFLICT (game_id) DO NOTHING;
    """, records)


def insert_player_game_stats(records):
    columns = ["player_id", "game_id", "team_id", *STAT_COLUMNS.values()]
    _insert(f"""
        INSERT INTO player_game_stats ({", ".join(columns)})
        VALUES ({", ".join(":" + c for c in columns)})
        ON CONFLICT (player_id, game_id) DO NOTHING;
    """, records)


def load_all(df):
    """Load in dependency order: referenced tables must exist first."""
    teams = build_teams(df)
    players = build_players(df)
    games = build_games(df)
    stats = build_player_game_stats(df)

    insert_teams(teams)
    print(f"Teams:             {len(teams):>7,}")
    insert_players(players)
    print(f"Players:           {len(players):>7,}")
    insert_games(games)
    print(f"Games:             {len(games):>7,}  "
          f"(neutral-site: {sum(g['neutral_site'] for g in games)})")
    insert_player_game_stats(stats)
    print(f"Player-game stats: {len(stats):>7,}")


if __name__ == "__main__":
    load_all(load_processed())
    print("All seasons loaded successfully!")
