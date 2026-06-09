from holidays.countries import Netherlands
from datetime import date
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame


class NetherlandsExtended(Netherlands):
    """
    Dutch public holidays + informal business closures:
    - Dec 26–31: post-Christmas / year-end shutdown

    Add further custom dates to `_extra_closures` as needed.
    """

    def _populate(self, year: int):
        super()._populate(year)  # load all official NL holidays first

        # Dec 26 is already 2nd Christmas Day in the base class;
        # add Dec 27–31 as informal closures
        self._extra_closures = {
            date(year, 12, 27): "Year-end closure",
            date(year, 12, 28): "Year-end closure",
            date(year, 12, 29): "Year-end closure",
            date(year, 12, 30): "Year-end closure",
            date(year, 12, 31): "New Year's Eve (informal closure)",
        }
        self.update(self._extra_closures)


nl_holidays = NetherlandsExtended()


def add_is_holiday(df: pd.DataFrame, dt_col: str | None = None) -> pd.DataFrame:
    dt = pd.to_datetime(df.index.to_series() if dt_col is None else df[dt_col])
    is_weekday = dt.dt.dayofweek < 5
    is_nl_holiday = dt.dt.date.map(lambda d: d in nl_holidays)

    df["is_holiday"] = (is_weekday & is_nl_holiday).astype(int)
    return df


def convert_to_ag_df(df: pd.DataFrame, time_col: str = "ds") -> pd.DataFrame:
    """Convert a df into the format expected by autogluon."""
    df["item_id"] = 1
    df.reset_index(inplace=True)
    ts = pd.to_datetime(df[time_col])
    df[time_col] = ts.dt.tz_convert(None) if ts.dt.tz is not None else ts
    df = TimeSeriesDataFrame(df, timestamp_column=time_col)

    return df


def slice_last_n_weeks(df: pd.DataFrame, n_weeks: int = None, time_col: str = "ds") -> pd.DataFrame:
    """
    Return the last N weeks of a dataframe.
    """
    ts = df.index.to_series() if df.index.name == time_col else df[time_col]
    cutoff = ts.max() - pd.Timedelta(weeks=n_weeks)
    return df[ts >= cutoff]