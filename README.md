# NBA Player Prediction Engine

Predicts an NBA player's **points, rebounds, assists, made threes and minutes**
for his next game, with an 80% prediction range, from five seasons of box
scores stored in PostgreSQL.

**Live dashboard:** https://ibraheem-s234.github.io/nba-player-prediction/

| | |
|---|---|
| Data | 131,291 player-games · 6,150 games · 1,029 players · 2021-22 to 2025-26 regular seasons |
| Features | 63 per player-game, every one computed only from earlier games |
| Model | Gradient-boosted trees (scikit-learn), one per statistic |
| Honest test | Trained on 2021-22 to 2024-25, scored once on all of 2025-26 |

## Results (2025-26 test season)

Mean absolute error: the average size of a miss, in the stat's own units.

| Stat | Season-average guess | Ridge | Random forest | **Gradient boosting** | Improvement | 80% range coverage |
|---|---:|---:|---:|---:|---:|---:|
| Points | 4.70 | 4.53 | 4.53 | **4.47** | 5.0% | 80.0% |
| Rebounds | 1.93 | 1.89 | 1.88 | **1.85** | 3.8% | 79.7% |
| Assists | 1.37 | 1.34 | 1.34 | **1.31** | 4.4% | 79.9% |
| 3-pointers made | 0.90 | 0.90 | 0.89 | **0.84** | 6.3% | 79.7% |
| Minutes | 5.32 | 4.74 | 4.67 | **4.62** | 13.1% | 80.1% |

**Main finding.** An ablation study (retraining while adding one group of
information at a time) shows that **lineup availability**, the minutes missing
from a team's usual rotation, is the most valuable context. It cut the points
error from 4.536 to 4.466 and the minutes error from 4.839 to 4.625, while
opponent defence cut points error by only 0.009. Who is missing from a
player's own team matters more than who he plays against.

## How it works

```
stats.nba.com ──nba_api──▶ data/raw/*.csv ──clean + validate──▶ data/processed/*.csv
                                                                      │
                                                                      ▼
                                            PostgreSQL: teams · players · games · player_game_stats
                                                                      │  (SQL JOIN query)
                                                                      ▼
                                            features: 63 leakage-safe inputs per player-game
                                                                      │
                                                                      ▼
                         train on 2021-25 ──▶ test on 2025-26 ──▶ retrain on all ──▶ projections
                                                                      │
                                                                      ▼
                                                    docs/ static dashboard (GitHub Pages)
```

1. **Ingestion** (`src/ingestion/nba_stats.py`) downloads each season's player
   game logs, keeps 31 useful columns, keeps `GAME_ID` as text (its leading
   zeros matter), derives `HOME` and `OPPONENT`, and refuses to save data with
   duplicates, missing keys, or games without exactly two teams.
2. **Database** (`src/database/`) is a normalized PostgreSQL schema. Foreign
   keys tie every box-score line to a real player, game and team. A `CHECK`
   constraint allows a missing home team only for the 10 neutral-site games
   (Mexico City, Paris, NBA Cup semifinals). A `team_game_totals` view rolls
   player lines up to team totals. Loading is idempotent: re-running never
   creates duplicates.
3. **Features** (`src/features/build_features.py`) are built in five groups:
   player form (3/5/10-game averages, season averages, volatility, usage),
   game context (home, rest, back-to-backs), opponent defence and pace, own-team
   offence, and lineup availability.
4. **Models** (`src/models/train.py`) compare two no-ML baselines, ridge
   regression, a random forest and gradient boosting. Settings are chosen on
   2024-25. The final score comes from 2025-26, which is used once.
5. **Prediction ranges** (`src/models/intervals.py`) use split conformal
   prediction scaled by each player's volatility, calibrated on 2024-25, so
   streaky players get wider ranges.
6. **Projections** (`src/predictions/predict.py`) reuse the exact feature code:
   an empty "next game" row is appended and run through the same pipeline,
   then priced against all 29 opponents, home and away.

### Preventing data leakage

A prediction for game N may only use games 1 to N−1. Every rolling statistic
applies `shift(1)` before averaging, and seasons are split by time, never at
random. `tests/test_features.py` enforces this on real data: it rewrites the
results of one date's games, rebuilds all 63 features, and fails if any
feature on or before that date changes. Removing a single `shift(1)` makes the
test fail.

### Design decisions

- **Neutral-site games keep no home team** instead of an invented one, so the
  model is never told about a home-court advantage that did not exist.
- **Players are identified by ID, not name.** Twelve players changed spelling
  across seasons (for example "Jonas Valanciunas" / "Jonas Valančiūnas").
- **Median, not mean.** The model minimises absolute error, so it predicts the
  median outcome, which is the right target when the metric is MAE.
- **No psychological or medical guessing.** Observable workload (rest,
  back-to-backs, minutes) stands in for fatigue.

## Run it yourself

Requirements: Python 3.12, PostgreSQL.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env            # then put your PostgreSQL password in .env
$env:PYTHONPATH = (Get-Location).Path

python -m src.ingestion.nba_stats              # download + clean all seasons (skips existing files)
python -m src.database.schema                  # create tables (add --reset to rebuild them)
python -m src.database.load_data               # load every season into PostgreSQL
python -m src.features.build_features          # features from PostgreSQL (--source csv works without a DB)
python -m src.models.train                     # compare models, backtest, save final models (~10 min)
python -m src.predictions.predict              # projections + dashboard data in docs/data/
python -m unittest discover -s tests -v        # 17 tests, including the leakage test
```

Open the dashboard locally with `python -m http.server --directory docs` and
visit http://localhost:8000.

## Project structure

```
src/
  config.py                  seasons, paths, settings (the one place to change them)
  ingestion/nba_stats.py     download, clean, validate
  database/connection.py     SQLAlchemy engine from .env
  database/schema.py         tables, constraints, indexes, view
  database/load_data.py      CSV -> PostgreSQL (pure build_* + batch insert_*)
  database/queries.py        PostgreSQL -> DataFrame for modelling
  features/build_features.py the 63 features
  models/evaluate.py         MAE, RMSE, R², coverage
  models/intervals.py        calibrated 80% ranges
  models/train.py            comparison, backtest, ablation, importance
  predictions/predict.py     next-game projections + dashboard export
tests/                       unit tests and early exploration scripts
docs/                        the static dashboard
reports/                     metrics.json and 2025-26 backtest predictions
```

## Limitations

- Box scores list only players who played, so absences are inferred after the
  fact. In live use they would come from the official inactive list, which is
  released before tip-off.
- Dashboard projections carry each player's 2025-26 form and team forward, so
  they cannot see off-season moves.
- No betting-market, play-by-play or player-tracking data is used.

Built with Python, pandas, scikit-learn, PostgreSQL, SQLAlchemy and nba_api.
