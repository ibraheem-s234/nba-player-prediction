"""
Tests for src/features/build_features.py.

Run all tests from the project root:
    python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
import pandas as pd

from src.config import SEASONS
from src.features.build_features import ALL_FEATURES, build_features
from src.ingestion.nba_stats import load_processed


def make_game(game_id, date, season, home_team, away_team, home_players, away_players):
    """Build the box-score rows of one synthetic game.
    *_players: list of (player_id, minutes, points)."""
    rows = []
    for team, opp, is_home, players in [
        (home_team, away_team, 1, home_players),
        (away_team, home_team, 0, away_players),
    ]:
        for player_id, minutes, points in players:
            rows.append({
                "SEASON_YEAR": season, "PLAYER_ID": player_id,
                "PLAYER_NAME": f"P{player_id}", "TEAM_ID": team,
                "TEAM_ABBREVIATION": f"T{team}", "GAME_ID": game_id,
                "GAME_DATE": pd.Timestamp(date), "HOME": is_home,
                "OPPONENT": f"T{opp}", "MIN": float(minutes), "PTS": points,
                "REB": 5, "AST": 3, "FG3M": 1, "FGA": 10, "FG3A": 4, "FTA": 2,
                "OREB": 1, "TOV": 1, "STL": 1, "BLK": 0,
            })
    return rows


class SyntheticFeatureTests(unittest.TestCase):
    """Small hand-made data where every correct answer is known."""

    @classmethod
    def setUpClass(cls):
        rows = []
        points = [10, 20, 30, 40, 50, 60]
        for i, pts in enumerate(points):
            rows += make_game(
                game_id=f"00221000{i:02d}", date=f"2021-11-{i * 2 + 1:02d}",
                season="2021-22", home_team=1, away_team=2,
                # Player 101 plays every game; player 102 sits out game index 4.
                home_players=[(101, 30, pts)] + ([] if i == 4 else [(102, 20, 8)]),
                away_players=[(201, 30, 15)],
            )
        cls.features = build_features(pd.DataFrame(rows))
        cls.p101 = (cls.features[cls.features["PLAYER_ID"] == 101]
                    .sort_values("GAME_DATE").reset_index(drop=True))

    def test_first_game_has_no_history(self):
        self.assertTrue(np.isnan(self.p101.loc[0, "PTS_L3"]))
        self.assertEqual(self.p101.loc[0, "GAMES_PLAYED_PRIOR"], 0)

    def test_rolling_average_uses_only_previous_games(self):
        # Game index 3 (40 pts): previous three games were 10, 20, 30.
        self.assertAlmostEqual(self.p101.loc[3, "PTS_L3"], 20.0)
        # Game index 5 (60 pts): previous three were 30, 40, 50.
        self.assertAlmostEqual(self.p101.loc[5, "PTS_L3"], 40.0)
        # Fewer than 10 earlier games: average of what exists (10..50).
        self.assertAlmostEqual(self.p101.loc[5, "PTS_L10"], 30.0)

    def test_season_average_excludes_current_game(self):
        self.assertTrue(np.isnan(self.p101.loc[0, "PTS_SEASON_AVG"]))
        self.assertAlmostEqual(self.p101.loc[2, "PTS_SEASON_AVG"], 15.0)

    def test_rest_days(self):
        self.assertEqual(self.p101.loc[1, "REST_DAYS"], 2)

    def test_opponent_allowed_points_use_previous_games(self):
        # Team 2 allowed (player 101 + player 102) points in each game.
        # Before game index 2 it had allowed 10+8=18 and 20+8=28 -> 23.
        self.assertAlmostEqual(self.p101.loc[2, "OPP_ALLOWED_PTS_L10"], 23.0)

    def test_missing_minutes_detects_absent_teammate(self):
        # Player 102 (20 minutes per game) sits out game index 4.
        self.assertAlmostEqual(self.p101.loc[4, "TEAM_MISSING_MIN"], 20.0)
        self.assertAlmostEqual(self.p101.loc[3, "TEAM_MISSING_MIN"], 0.0)


class LeakageTest(unittest.TestCase):
    """
    The decisive test on REAL data: change the results of one date's games
    and rebuild every feature. If any feature on or before that date
    changes, information from the game leaked into its own prediction.
    """

    def test_changing_a_game_never_changes_its_own_features(self):
        try:
            logs = load_processed([SEASONS[0]])
        except FileNotFoundError:
            self.skipTest("processed data not found; run the ingestion step first")

        dates = sorted(logs["GAME_DATE"].unique())
        cutoff = dates[60]                       # a date in mid-season

        changed = logs.copy()
        on_cutoff = changed["GAME_DATE"] == cutoff
        for stat in ["PTS", "REB", "AST", "FG3M", "MIN", "FGA", "FTA", "TOV"]:
            changed.loc[on_cutoff, stat] = changed.loc[on_cutoff, stat] * 3 + 7

        before = build_features(logs).set_index(["PLAYER_ID", "GAME_ID"])
        after = build_features(changed).set_index(["PLAYER_ID", "GAME_ID"])

        up_to_cutoff = before["GAME_DATE"] <= cutoff
        pd.testing.assert_frame_equal(
            before.loc[up_to_cutoff, ALL_FEATURES],
            after.loc[up_to_cutoff, ALL_FEATURES],
        )
        # Sanity check that the change DID reach later games.
        later = before["GAME_DATE"] > cutoff
        self.assertFalse(before.loc[later, "PTS_L3"].equals(after.loc[later, "PTS_L3"]))


if __name__ == "__main__":
    unittest.main()
