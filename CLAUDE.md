# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Forecasts the hourly CO₂ emission factor for the Dutch national grid 7 days ahead, using renewable energy production forecasts from the [NED API](https://api.ned.nl) plus weather data from [Open-Meteo](https://open-meteo.com). The target variable is `emissionfactor` (gCO₂/kWh), which depends on how much of total grid production comes from renewables (solar, onshore wind, offshore wind).

## Setup

Uses [Poetry](https://python-poetry.org/) with Python 3.12. A `.env` file with `NED_API_KEY` is required.

```bash
pyenv local 3.12.12
pip install poetry
poetry install
```

## Commands

```bash
# Run all tests
poetry run pytest

# Run a single test
poetry run pytest tests/test_utils.py::test_slice_last_n_weeks

# Train the direct model (predicts emissionfactor directly)
poetry run python -m src.model_trainer   # or run indirect_model.py as a script

# Run inference (generates 7-day forecast, saves to artifacts/data/predictions/)
poetry run python inference.py
```

## Architecture

There are two model approaches, both using AutoGluon TimeSeries as the forecasting backbone:

**Direct model** (`ModelTrainer.train_direct_model` → `inference.py`): Predicts `emissionfactor` directly from exogenous features (renewable volume, temperature, holiday flag). This is the production path — `inference.py` loads from `artifacts/models/direct/`.

**Indirect model** (`indirect_model.py`): Two-step approach — first predicts total grid volume with AutoGluon, then derives the emission factor via OLS regression on `fraction_renewable`. Experimental/validation only.

### Key classes and their roles

- **`DataManager`** (`src/data_manager.py`): All data I/O. Fetches NED production data (historical and forecast), temperature from Open-Meteo, and assembles the final training DataFrame. `prepare_train_test_df(n_weeks)` is the main entrypoint for training data.
- **`ModelTrainer`** (`src/model_trainer.py`): Wraps AutoGluon `TimeSeriesPredictor`. `ModelConfig` dataclass holds all hyperparameters. `find_optimal_training_window` compares candidate dataset lengths and picks the best-scoring one.
- **`utils.py`** (`src/utils.py`): `convert_to_ag_df` reformats DataFrames for AutoGluon (adds `item_id=1`, strips timezone). `slice_last_n_weeks` slices training windows. `NetherlandsExtended` adds Dec 27–31 year-end closures on top of official NL public holidays.

### Data flow (training)

```
NED API (historical) ──┐
                        ├─► DataManager.get_last_n_weeks()
Open-Meteo (archive) ──┘         │
                                  ▼
                         add_exo_features()
                         (holiday, temp, renewable totals)
                                  │
                                  ▼
                         train_test_split()
                                  │
                         ┌────────┴────────┐
                         ▼                 ▼
                      train_df           test_df (last week)
                         │
                  find_optimal_training_window()
                         │
                  fit_predictor() ──► artifacts/models/direct/
```

### Data flow (inference)

```
NED API (forecast) ──┐
                      ├─► forecast covariates for next 7 days
Open-Meteo (fcst) ───┘
                              │
NED API (historical) ─────► historical context window
                              │
                      predictor.predict() ──► artifacts/data/predictions/latest_preds.csv
```

### NED API notes

- Date args use `DD-MM-YYYY` for `get_ned_production_data` internally but `YYYY-MM-DD` for temperature calls.
- The API is paginated at 200 items/page; the client handles pagination automatically.
- `ENERGY_TYPE_MAP` and `CLASSIFICATION_MAP` are `bidict` instances, so they support reverse lookups (used in `prepare_ned_data` to recover the source name from the returned type code).

### Artifacts layout

```
artifacts/
  models/
    direct/          # AutoGluon predictor + train_config.json (best_train_window)
    indirect/        # AutoGluon predictor for total volume step
  data/
    train-test/      # Saved CSVs from training runs
    predictions/     # latest_preds.csv from inference.py
```
