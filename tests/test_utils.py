import pandas as pd
import pytest
from src.utils import slice_last_n_weeks


def test_slice_last_n_weeks():
    # 4 weeks of hourly data (4 * 7 * 24 = 672 rows), n_weeks=4 should return all rows
    dates = pd.date_range(start="2024-01-01", periods=4 * 7 * 24, freq="h")
    df = pd.DataFrame({"ds": dates, "value": range(len(dates))})

    result = slice_last_n_weeks(df, n_weeks=4, time_col="ds")

    assert len(result) == len(df)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))
