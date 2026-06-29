import pandas as pd
from src.data_manager import DataManager


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
