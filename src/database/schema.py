"""
Database schema: creates the PostgreSQL tables (and one view).

Run from the project root:
    python -m src.database.schema            # create anything missing (safe to repeat)
    python -m src.database.schema --reset    # DROP all project tables, then recreate

Tables, in dependency order (a table can only reference tables above it):

    teams              one row per NBA team
    players            one row per player
    games              one row per game (home/away, or neutral site)
    player_game_stats  one row per player per game (the box score line)

    team_game_totals   a VIEW: player rows summed into one row per team per game
"""

import argparse

# Each statement is kept as plain SQL so it can be read (and tested) on its own.
CREATE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS teams (
        team_id      INTEGER PRIMARY KEY,
        team_name    VARCHAR(100) NOT NULL,
        abbreviation VARCHAR(5)   NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS players (
        player_id   INTEGER PRIMARY KEY,
        player_name VARCHAR(100) NOT NULL
    );
    """,
    # A neutral-site game has no home team, so home/away are allowed to be
    # NULL -- but only for neutral games. The CHECK constraint makes the
    # database itself enforce that rule.
    """
    CREATE TABLE IF NOT EXISTS games (
        game_id      VARCHAR(10) PRIMARY KEY,
        season       VARCHAR(7)  NOT NULL,
        game_date    DATE        NOT NULL,
        home_team_id INTEGER REFERENCES teams(team_id),
        away_team_id INTEGER REFERENCES teams(team_id),
        neutral_site BOOLEAN     NOT NULL DEFAULT FALSE,
        CONSTRAINT games_home_away_check CHECK (
            (neutral_site AND home_team_id IS NULL AND away_team_id IS NULL)
            OR
            (NOT neutral_site AND home_team_id IS NOT NULL
                AND away_team_id IS NOT NULL
                AND home_team_id <> away_team_id)
        )
    );
    """,
    # team_id records which team the player played for IN THIS GAME.
    # Players get traded, so a player's team belongs to the game, not the player.
    """
    CREATE TABLE IF NOT EXISTS player_game_stats (
        player_id                INTEGER     NOT NULL REFERENCES players(player_id),
        game_id                  VARCHAR(10) NOT NULL REFERENCES games(game_id),
        team_id                  INTEGER     NOT NULL REFERENCES teams(team_id),
        minutes                  DECIMAL(5,2),
        field_goals_made         INTEGER,
        field_goals_attempted    INTEGER,
        field_goal_percentage    DECIMAL(5,3),
        three_pointers_made      INTEGER,
        three_pointers_attempted INTEGER,
        three_point_percentage   DECIMAL(5,3),
        free_throws_made         INTEGER,
        free_throws_attempted    INTEGER,
        free_throw_percentage    DECIMAL(5,3),
        offensive_rebounds       INTEGER,
        defensive_rebounds       INTEGER,
        rebounds                 INTEGER,
        assists                  INTEGER,
        steals                   INTEGER,
        blocks                   INTEGER,
        turnovers                INTEGER,
        personal_fouls           INTEGER,
        points                   INTEGER,
        plus_minus               INTEGER,
        PRIMARY KEY (player_id, game_id)
    );
    """,
    # Indexes make the most common lookups fast: "all games for a player",
    # "all players in a game", and "games in date order".
    "CREATE INDEX IF NOT EXISTS idx_stats_game ON player_game_stats (game_id);",
    "CREATE INDEX IF NOT EXISTS idx_stats_team ON player_game_stats (team_id);",
    "CREATE INDEX IF NOT EXISTS idx_games_date ON games (game_date);",
    # A view is a saved query that behaves like a read-only table. This one
    # rolls player lines up to team totals, which the opponent-defence
    # features are built from.
    """
    CREATE OR REPLACE VIEW team_game_totals AS
    SELECT
        s.game_id,
        g.season,
        g.game_date,
        s.team_id,
        SUM(s.points)                AS points,
        SUM(s.rebounds)              AS rebounds,
        SUM(s.assists)               AS assists,
        SUM(s.three_pointers_made)   AS three_pointers_made,
        SUM(s.field_goals_attempted) AS field_goals_attempted,
        SUM(s.free_throws_attempted) AS free_throws_attempted,
        SUM(s.turnovers)             AS turnovers,
        SUM(s.offensive_rebounds)    AS offensive_rebounds
    FROM player_game_stats AS s
    JOIN games AS g ON g.game_id = s.game_id
    GROUP BY s.game_id, g.season, g.game_date, s.team_id;
    """,
]

# Reverse dependency order: drop things that reference others first.
DROP_STATEMENTS = [
    "DROP VIEW IF EXISTS team_game_totals;",
    "DROP TABLE IF EXISTS player_game_stats;",
    "DROP TABLE IF EXISTS games;",
    "DROP TABLE IF EXISTS players;",
    "DROP TABLE IF EXISTS teams;",
]


def _execute_all(statements):
    # Imported inside the function so this file can be read/tested
    # without a database connection.
    from sqlalchemy import text
    from src.database.connection import engine

    # engine.begin() opens ONE transaction: either every statement
    # succeeds, or none of them are applied.
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def create_tables():
    _execute_all(CREATE_STATEMENTS)


def drop_tables():
    _execute_all(DROP_STATEMENTS)


def main():
    parser = argparse.ArgumentParser(description="Create the database schema.")
    parser.add_argument("--reset", action="store_true",
                        help="Drop all project tables first (deletes their data).")
    args = parser.parse_args()

    if args.reset:
        answer = input("This DELETES all rows in teams, players, games and "
                       "player_game_stats. Type 'reset' to continue: ")
        if answer.strip() != "reset":
            print("Cancelled. Nothing was changed.")
            return
        drop_tables()
        print("Old tables dropped.")

    create_tables()
    print("Database tables created successfully!")


if __name__ == "__main__":
    main()
