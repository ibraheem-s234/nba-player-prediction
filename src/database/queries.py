"""
Read data back OUT of PostgreSQL for the modelling pipeline.

The database is the project's system of record: features are built from
what this query returns. It rebuilds the same columns the processed CSVs
have (HOME, OPPONENT, ...) using JOINs, so the feature code works the same
whether the data came from the database or from the CSV files.
"""

import pandas as pd

GAME_LOGS_SQL = """
WITH game_teams AS (
    -- the two teams that played in each game
    SELECT DISTINCT game_id, team_id FROM player_game_stats
),
opponents AS (
    -- pair each team with the OTHER team in the same game
    SELECT a.game_id, a.team_id, b.team_id AS opponent_team_id
    FROM game_teams AS a
    JOIN game_teams AS b
      ON a.game_id = b.game_id AND a.team_id <> b.team_id
)
SELECT
    g.season                       AS "SEASON_YEAR",
    s.player_id                    AS "PLAYER_ID",
    p.player_name                  AS "PLAYER_NAME",
    s.team_id                      AS "TEAM_ID",
    t.abbreviation                 AS "TEAM_ABBREVIATION",
    s.game_id                      AS "GAME_ID",
    g.game_date                    AS "GAME_DATE",
    CASE WHEN g.home_team_id = s.team_id THEN 1 ELSE 0 END AS "HOME",
    CASE WHEN g.neutral_site THEN 1 ELSE 0 END             AS "NEUTRAL_SITE",
    o.opponent_team_id             AS "OPPONENT_TEAM_ID",
    ot.abbreviation                AS "OPPONENT",
    s.minutes                      AS "MIN",
    s.points                       AS "PTS",
    s.rebounds                     AS "REB",
    s.offensive_rebounds           AS "OREB",
    s.assists                      AS "AST",
    s.three_pointers_made          AS "FG3M",
    s.three_pointers_attempted     AS "FG3A",
    s.field_goals_attempted        AS "FGA",
    s.free_throws_attempted        AS "FTA",
    s.turnovers                    AS "TOV",
    s.steals                       AS "STL",
    s.blocks                       AS "BLK"
FROM player_game_stats AS s
JOIN players   AS p  ON p.player_id = s.player_id
JOIN games     AS g  ON g.game_id   = s.game_id
JOIN teams     AS t  ON t.team_id   = s.team_id
JOIN opponents AS o  ON o.game_id   = s.game_id AND o.team_id = s.team_id
JOIN teams     AS ot ON ot.team_id  = o.opponent_team_id
ORDER BY g.game_date, s.game_id, s.player_id;
"""


def read_game_logs_from_db():
    """Return every player-game as a DataFrame, straight from PostgreSQL."""
    from src.database.connection import engine

    df = pd.read_sql(GAME_LOGS_SQL, engine)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN"] = df["MIN"].astype(float)  # DECIMAL arrives as Python Decimal
    return df
