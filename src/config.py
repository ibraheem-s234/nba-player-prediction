"""
Central project configuration.

Every path, season list, and modelling constant lives here so the rest of
the code never hard-codes them. If you want to add a season, this is the
only file you change.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/config.py -> parents[0] is src/, parents[1] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # untouched API downloads
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned, one CSV per season
FEATURES_PATH = PROCESSED_DIR / "features.csv"  # model-ready dataset

MODELS_DIR = PROJECT_ROOT / "models"    # trained model files (.joblib)
REPORTS_DIR = PROJECT_ROOT / "reports"  # metrics and evaluation outputs
SITE_DIR = PROJECT_ROOT / "docs"        # the static dashboard (GitHub Pages serves /docs)
SITE_DATA_DIR = SITE_DIR / "data"       # JSON the dashboard reads

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------

# Regular seasons we collect, oldest first. Order matters: when the same
# player appears with two spellings, the most recent season wins.
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

# Time-based split. The model never sees a game from a later season while
# learning, which mirrors how it would be used in real life.
TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24"]
VALIDATION_SEASON = "2024-25"   # used to choose model settings
TEST_SEASON = "2025-26"         # touched once, for the final report


def season_file_name(season: str) -> str:
    """'2024-25' -> 'player_game_logs_2024_25.csv' (matches the original file)."""
    return f"player_game_logs_{season.replace('-', '_')}.csv"


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------

# Statistics we predict. Keys are column names in the data,
# values are labels for the dashboard.
TARGETS = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "FG3M": "3-Pointers Made",
    "MIN": "Minutes",
}

# A row only becomes a training/test example once the player has this many
# earlier games, so every example has a meaningful recent-form history.
MIN_PRIOR_GAMES = 5

# ---------------------------------------------------------------------------
# Next-game projections (what the dashboard shows)
# ---------------------------------------------------------------------------

# The projected game is "the next game, as if the latest season continued":
# the player's end-of-season form and season averages carry over. It is
# dated this many days after the season's final game (normal rest).
NEXT_GAME_DAYS_AFTER_SEASON = 2

# Players included in the projections: they played at least this many games
# in the latest season and average at least this many minutes recently.
PROJECTION_MIN_GAMES = 10
PROJECTION_MIN_MINUTES = 10

RANDOM_STATE = 42
