import pandas as pd
from pathlib import Path
from src.data_manager import DataManager

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_get_last_n_weeks():
    dm = DataManager()
    df = dm.get_last_n_weeks(n_weeks=1, sources=("total", "solar"))

    for source in ("total", "solar"):
        assert f"volume_{source}" in df.columns

    assert len(df) == 24 * 7


def test_get_forecast_data():
    dm = DataManager()
    df = dm.get_forecast_data(sources=("solar", "offshore_wind"))

    for source in ("solar", "offshore_wind"):
        assert f"volume_{source}" in df.columns

    assert len(df) == 24 * 7


def test_train_test_split():
    dates = pd.date_range(start="2026-06-15", periods=2 * 7 * 24, freq="h")
    df = pd.DataFrame({"value": range(len(dates))}, index=dates)

    dm = DataManager()
    train_df, test_df = dm.train_test_split(df=df, train_test_split_date="2026-06-22")

    assert test_df.index.min() - train_df.index.max() == pd.Timedelta(hours=1)
    assert len(train_df) + len(test_df) == len(df)


def test_add_exo_features():
    ned_last_week = pd.read_csv(FIXTURES_DIR / "ned_last_week.csv", index_col="ds", parse_dates=True)
    start_date = ned_last_week.index.min().strftime("%Y-%m-%d")
    end_date = ned_last_week.index.max().strftime("%Y-%m-%d")

    result = DataManager.add_exo_features(
        df=ned_last_week,
        mode="historical",
        start_date=start_date,
        end_date=end_date,
    )

    assert result.isnull().sum().sum() == 0
