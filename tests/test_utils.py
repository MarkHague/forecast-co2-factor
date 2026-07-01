import pandas as pd
import pytest
from autogluon.timeseries import TimeSeriesDataFrame
from src.utils import slice_last_n_weeks, convert_to_ag_df, add_is_holiday


def test_slice_last_n_weeks():
    # 4 weeks of hourly data (4 * 7 * 24 = 672 rows), n_weeks=4 should return all rows
    dates = pd.date_range(start="2024-01-01", periods=4 * 7 * 24, freq="h")
    df = pd.DataFrame({"ds": dates, "value": range(len(dates))})

    result = slice_last_n_weeks(df, n_weeks=4, time_col="ds")

    assert len(result) == len(df)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))


def test_convert_to_ag_df():
    dates = pd.date_range("2026-06-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"emissionfactor": range(24)}, index=dates)
    df.index.name = "ds"

    result = convert_to_ag_df(df)

    assert isinstance(result, TimeSeriesDataFrame)
    assert (result.index.get_level_values("item_id") == 1).all()
    assert result.index.get_level_values("timestamp").tz is None


def test_add_is_holiday():
    dates = pd.to_datetime([
        "2025-05-05",  # Liberation Day (Monday) → 1
        "2025-05-06",  # Regular Tuesday → 0
        "2025-12-25",  # Christmas Day (Thursday) → 1
        "2025-12-27",  # Year-end closure (Saturday) → 0, weekend overrides
        "2025-12-29",  # Year-end closure (Monday) → 1
        "2025-12-31",  # New Year's Eve closure (Wednesday) → 1
        "2026-01-01",  # New Year's Day (Thursday) → 1
    ])
    df = pd.DataFrame(index=dates)

    result = add_is_holiday(df)

    expected = [1, 0, 1, 0, 1, 1, 1]
    assert result["is_holiday"].tolist() == expected
