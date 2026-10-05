"""
Tests for data cleaning (src/ingestion/nba_stats.py) and the database
record builders (src/database/load_data.py). No database or network needed.
"""

import unittest

import pandas as pd

from src.database.load_data import (build_games, build_player_game_stats,
                                    build_players, build_teams)
from src.ingestion.nba_stats import RAW_COLUMNS, clean_game_logs


def raw_row(player_id, name, team_id, abbr, game_id, date, matchup, pts=10):
    row = {c: 0 for c in RAW_COLUMNS}
    row.update({
        "SEASON_YEAR": "2024-25", "PLAYER_ID": player_id, "PLAYER_NAME": name,
        "TEAM_ID": team_id, "TEAM_ABBREVIATION": abbr, "TEAM_NAME": f"Team {abbr}",
        "GAME_ID": game_id, "GAME_DATE": date, "MATCHUP": matchup, "WL": "W",
        "MIN": 30.5, "PTS": pts, "FG_PCT": 0.5, "FG3_PCT": 0.4, "FT_PCT": 0.8,
    })
    return row


def sample_raw():
    return pd.DataFrame([
        # Normal game: GSW at home against LAL.
        raw_row(1, "Home Guy", 100, "GSW", 22400001, "2024-10-22T00:00:00", "GSW vs. LAL"),
        raw_row(2, "Away Guy", 200, "LAL", 22400001, "2024-10-22T00:00:00", "LAL @ GSW"),
        # Neutral-site game: both sides written with "@".
        raw_row(1, "Home Guy", 100, "GSW", 22400002, "2024-11-02T00:00:00", "GSW @ LAL"),
        raw_row(2, "Away Guy", 200, "LAL", 22400002, "2024-11-02T00:00:00", "LAL @ GSW"),
    ])


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.clean = clean_game_logs(sample_raw())

    def test_game_id_keeps_leading_zeros(self):
        self.assertEqual(set(self.clean["GAME_ID"]), {"0022400001", "0022400002"})

    def test_home_and_opponent(self):
        first = self.clean[self.clean["GAME_ID"] == "0022400001"].set_index("PLAYER_ID")
        self.assertEqual(first.loc[1, "HOME"], 1)
        self.assertEqual(first.loc[2, "HOME"], 0)
        self.assertEqual(first.loc[1, "OPPONENT"], "LAL")

    def test_dates_are_real_dates(self):
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.clean["GAME_DATE"]))

    def test_duplicates_are_rejected(self):
        raw = pd.concat([sample_raw(), sample_raw().iloc[[0]]])
        with self.assertRaises(ValueError):
            clean_game_logs(raw)

    def test_game_with_one_team_is_rejected(self):
        with self.assertRaises(ValueError):
            clean_game_logs(sample_raw().iloc[[0]])


class RecordBuilderTests(unittest.TestCase):
    def setUp(self):
        self.clean = clean_game_logs(sample_raw())

    def test_games_normal_and_neutral(self):
        games = {g["game_id"]: g for g in build_games(self.clean)}
        normal, neutral = games["0022400001"], games["0022400002"]
        self.assertEqual((normal["home_team_id"], normal["away_team_id"]), (100, 200))
        self.assertFalse(normal["neutral_site"])
        self.assertIsNone(neutral["home_team_id"])
        self.assertTrue(neutral["neutral_site"])
        self.assertEqual(normal["season"], "2024-25")

    def test_latest_player_name_wins(self):
        renamed = self.clean.copy()
        later = renamed["GAME_ID"] == "0022400002"
        renamed.loc[later & (renamed["PLAYER_ID"] == 1), "PLAYER_NAME"] = "Home Guy Jr."
        names = {p["player_id"]: p["player_name"] for p in build_players(renamed)}
        self.assertEqual(names[1], "Home Guy Jr.")

    def test_one_team_record_per_team(self):
        self.assertEqual(len(build_teams(self.clean)), 2)

    def test_stats_records_have_plain_python_types(self):
        record = build_player_game_stats(self.clean)[0]
        self.assertIsInstance(record["player_id"], int)
        self.assertIsInstance(record["minutes"], float)
        self.assertIsInstance(record["points"], int)
        self.assertEqual(len(record["game_id"]), 10)


if __name__ == "__main__":
    unittest.main()
