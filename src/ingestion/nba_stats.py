"""
Data ingestion: download NBA player game logs and clean them.

Pipeline for each season:
    stats.nba.com  --(nba_api)-->  data/raw/<season>.csv        (untouched values)
                   --clean()---->  data/processed/<season>.csv  (typed + derived columns)

Run from the project root:
    python -m src.ingestion.nba_stats                   # every season in config.SEASONS
    python -m src.ingestion.nba_stats --seasons 2025-26 # just one season
    python -m src.ingestion.nba_stats --process-only    # re-clean raw files, no network
"""

import argparse
import time

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, SEASONS, season_file_name

# The 31 columns we keep from the 70+ the API returns. The *_RANK columns and
# fantasy columns are summaries computed by the NBA, not information we want
# feeding the model.
RAW_COLUMNS = [
    "SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PFD", "PTS",
    "PLUS_MINUS",
]

REQUEST_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 3


def _request_game_logs(season, date_from="", date_to=""):
    """One call to the PlayerGameLogs endpoint, with a timeout."""
    # Imported here so --process-only works even without network packages.
    from nba_api.stats.endpoints import playergamelogs

    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable="Regular Season",
        date_from_nullable=date_from,
        date_to_nullable=date_to,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return logs.get_data_frames()[0]


def _with_retries(description, func, *args):
    """Call func(*args), retrying with a growing pause if the server fails."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return func(*args)
        except Exception as error:  # network errors, timeouts, HTTP 500s
            print(f"[Ingestion:nba_stats] {description} failed "
                  f"(attempt {attempt}/{MAX_ATTEMPTS}): {error}")
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(10 * attempt)


def fetch_player_game_logs(season):
    """
    Download every regular-season player game log for one season.

    stats.nba.com sometimes answers a full-season request with an HTTP 500
    error, while smaller requests succeed. So if the full season fails we
    fall back to requesting it in date ranges and joining the pieces.
    """
    print(f"Fetching NBA player game logs for {season}...")
    try:
        df = _with_retries(f"{season} full season", _request_game_logs, season)
    except Exception:
        print(f"[Ingestion:nba_stats] Falling back to date-range requests for {season}.")
        start_year = int(season[:4])
        ranges = [
            (f"10/01/{start_year}", f"11/30/{start_year}"),
            (f"12/01/{start_year}", f"01/31/{start_year + 1}"),
            (f"02/01/{start_year + 1}", f"04/30/{start_year + 1}"),
        ]
        parts = [
            _with_retries(f"{season} {a}-{b}", _request_game_logs, season, a, b)
            for a, b in ranges
        ]
        df = pd.concat(parts, ignore_index=True)
        # Ranges are non-overlapping, but guard against duplicates anyway.
        df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"])

    return df[RAW_COLUMNS]


def clean_game_logs(df):
    """
    Turn raw game logs into the processed format.

    - GAME_ID stays text with its leading zeros ("0022401198"). Read as a
      number it would silently become 22401198.
    - GAME_DATE becomes a real date.
    - HOME = 1 when the matchup reads "XXX vs. YYY" (home), 0 for "XXX @ YYY".
      Neutral-site games list BOTH teams with "@", so both get HOME = 0.
    - OPPONENT = the three-letter code at the end of MATCHUP.
    Then the data is validated. If anything is wrong we stop with an error
    instead of saving bad data.
    """
    df = df[RAW_COLUMNS].copy()

    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    df["HOME"] = df["MATCHUP"].str.contains(" vs. ", regex=False).astype(int)
    df["OPPONENT"] = df["MATCHUP"].str[-3:]

    validate_game_logs(df)
    return df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID", "PLAYER_ID"])


def validate_game_logs(df):
    """Raise ValueError if the data breaks any rule we rely on later."""
    missing = df[["PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "MIN", "PTS"]].isna().sum()
    if missing.any():
        raise ValueError(f"Missing values in key columns:\n{missing[missing > 0]}")

    duplicates = df.duplicated(subset=["PLAYER_ID", "GAME_ID"]).sum()
    if duplicates:
        raise ValueError(f"{duplicates} duplicate player-game rows")

    teams_per_game = df.groupby("GAME_ID")["TEAM_ID"].nunique()
    if (teams_per_game != 2).any():
        bad = teams_per_game[teams_per_game != 2].index.tolist()[:5]
        raise ValueError(f"Games without exactly two teams, e.g. {bad}")

    if not df["MATCHUP"].str.contains(r" vs\. | @ ", regex=True).all():
        raise ValueError("Unexpected MATCHUP format found")


def read_raw(season):
    path = RAW_DIR / season_file_name(season)
    return pd.read_csv(path, dtype={"GAME_ID": str})


def process_season(season, download):
    """Download (optional), clean, and save one season. Returns the cleaned frame."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / season_file_name(season)

    if download or not raw_path.exists():
        raw = fetch_player_game_logs(season)
        raw.to_csv(raw_path, index=False)
        print(f"Saved raw data: {raw_path}")
    else:
        raw = read_raw(season)

    clean = clean_game_logs(raw)
    processed_path = PROCESSED_DIR / season_file_name(season)
    clean.to_csv(processed_path, index=False, date_format="%Y-%m-%d")

    print(f"{season}: {len(clean):,} player-games, "
          f"{clean['GAME_ID'].nunique():,} games, "
          f"{clean['PLAYER_ID'].nunique():,} players -> {processed_path.name}")
    return clean


def load_processed(seasons=None):
    """Read processed seasons back into one DataFrame (used by later stages)."""
    seasons = seasons or SEASONS
    frames = []
    for season in seasons:
        path = PROCESSED_DIR / season_file_name(season)
        frames.append(pd.read_csv(path, dtype={"GAME_ID": str}, parse_dates=["GAME_DATE"]))
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Download and clean NBA game logs.")
    parser.add_argument("--seasons", nargs="+", default=SEASONS,
                        help="Seasons like 2024-25 (default: all in config).")
    parser.add_argument("--process-only", action="store_true",
                        help="Only re-clean existing raw files; no downloading.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if a raw file already exists.")
    args = parser.parse_args()

    for season in args.seasons:
        if args.process_only and not (RAW_DIR / season_file_name(season)).exists():
            raise FileNotFoundError(f"No raw file for {season}; run without --process-only.")
        process_season(season, download=args.force and not args.process_only)


if __name__ == "__main__":
    main()
