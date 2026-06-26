import pandas as pd
import os, io
import math
import requests
from bidict import bidict
from dotenv import load_dotenv
import openmeteo_requests
import requests_cache
from retry_requests import retry

from src.utils import add_is_holiday


class DataManager:
    """Handles data retrieval, preprocessing and transformation."""

    ENERGY_TYPE_MAP = bidict({
        "total": 27,
        "offshore_wind": 17,
        "onshore_wind": 1,
        "solar": 2,
        "coal": 19
    } )

    CLASSIFICATION_MAP = bidict({
        "historical": 2,
        "forecast": 1
    })

    def __init__(self):
        load_dotenv()
        self.NED_API_KEY = os.getenv("NED_API_KEY")
        self.NED_API_URL = "https://api.ned.nl/v1/utilizations"

    def get_ned_production_data(self, start_date:str = None, end_date: str = None,
                                energy_type: str = "solar",
                                mode: str = "historical") -> pd.DataFrame:
        """
        Retrieves hourly production data from the NED API.

        Args:
            start_date: start date of requested time series
            end_date: end date of requested time series
            energy_type: energy source. Currently supported are "total", "solar", "onshore_wind", and "offshore_wind"
            mode:"historical" or "forecast"
        """

        df_out = []

        headers = {
            'X-AUTH-TOKEN': self.NED_API_KEY,
            'accept': 'text/csv'}

        # compute number of pages needed for request (must be less than 200)
        n_hours = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds() / 3600.0
        n_pages = math.ceil(n_hours / 200)

        if n_pages < 200: # the API can only process 200 requests every 5 minutes
            for page in range(1, n_pages + 1):
                params = {'point': 0,
                          'type': self.ENERGY_TYPE_MAP[energy_type],
                          'granularity': 5, # hourly
                          'granularitytimezone': 1,
                          'classification': self.CLASSIFICATION_MAP[mode],
                          'activity': 1,
                          'validfrom[strictly_before]': end_date,
                          'validfrom[after]': start_date,
                          'page': page,
                          'itemsPerPage': 200
                          }

                response = requests.get(self.NED_API_URL, headers=headers, params=params, allow_redirects=False)
                df = pd.read_csv(io.StringIO(response.text))
                df_out.append(df)

            df = pd.concat(df_out)

            return df
        else:
            print("Requested data exceeds API limits. Please reduce the requested length of the time series.")


    def train_test_split(self, df: pd.DataFrame = None,
                             train_test_split_date:str = None,
                             test_end_date: str = None) -> (pd.DataFrame, pd.DataFrame):
        """
        Split train and test sets. Data must be indexed with a valid datetime series.
        Note that validation data is included in the training data.

        Args:
            df: Input dataframe
            train_test_split_date: Date at which training data ends, and test data begins.
            test_end_date: Last timestamp of the test data.
        Returns:
            Tuple of train and test pandas Dataframes
        """

        split = pd.Timestamp(train_test_split_date)
        train_df = df[df.index <= split]
        test_df = df[df.index > split]
        if test_end_date:
            test_df = test_df[test_df.index <= pd.Timestamp(test_end_date)]

        return train_df, test_df

    def prepare_ned_data(self, df: pd.DataFrame = None):
        """
        Performs a few necessary preprocessing steps needing for data merging and training.
        """
        df = df.set_index(pd.to_datetime(df["validto"])).rename_axis('ds')
        type_code = int(df['type'].iloc[0][10:])
        source = self.ENERGY_TYPE_MAP.inverse[type_code]
        df.rename({'volume': f'volume_{source}'}, axis = 1, inplace=True)

        if source == "total":
            df = df[[f'volume_{source}','emissionfactor']]
        else:
            df = df[[f'volume_{source}']]

        return df

    def get_last_n_weeks(self, n_weeks: int = None,
                        sources: tuple[str] = ("total", "solar", "offshore_wind", "onshore_wind") ) -> pd.DataFrame:
        """
        Get historical NED data for all required energy sources, from N weeks ago to today.

        """
        dfs = []
        end_date = pd.Timestamp.now(tz='UTC')
        start_date = end_date - pd.Timedelta(weeks = n_weeks)
        for source in sources:
            df = self.get_ned_production_data(energy_type=source,
                                              start_date=start_date.strftime('%Y-%m-%d'),
                                              end_date=end_date.strftime('%Y-%m-%d'))
            df = self.prepare_ned_data(df = df)
            dfs.append(df)

        return pd.concat(dfs, axis=1)

    def get_forecast_data(self,
                          sources: tuple[str] = ("solar", "offshore_wind", "onshore_wind") ) -> pd.DataFrame:
        """
        Retrieves NED production forecast data.
        """
        dfs = []
        now = pd.Timestamp.now(tz='UTC')
        end_forecast = now + pd.Timedelta(days=7)

        for source in sources:
            df = self.get_ned_production_data(energy_type=source,
                                              mode = 'forecast',
                                              start_date=now.strftime('%Y-%m-%d'),
                                              end_date=end_forecast.strftime('%Y-%m-%d'))
            df = self.prepare_ned_data(df = df)
            dfs.append(df)

        return pd.concat(dfs, axis=1)


    @staticmethod
    def add_exo_features(df: pd.DataFrame = None, time_col: str = None,
                         mode: str = 'historical',
                         start_date: str = None,
                         end_date: str = None
                         ) -> pd.DataFrame:
        """
        Add exogenous features needed for model training.
        Args:
            df: Input data.
            time_col: column containing datetime information
            mode: Either 'historical' or 'forecast', used for retrieving temperature data
            start_date: Start date of time series in format YYYY-MM-DD, used for retrieving temperature data. Ignored if mode = 'forecast'.
            end_date: End date of time series in format YYYY-MM-DD, used for retrieving temperature data. Ignored if mode = 'forecast'.
        Features added:
            - "is_holiday", Dutch holidays, plus period xmas - new year
            - "volume_total_renewable", sum of wind and solar sources
            - "fraction_renewable", fraction of total production coming from renewables
            - "temperature", historical or forecasted air temperature

        """
        df = add_is_holiday(df, dt_col=time_col)
        # compute total renewable volume, drop individual sources
        df["volume_total_renewable"] = df["volume_solar"] + df["volume_onshore_wind"] + df["volume_offshore_wind"]
        df.drop(["volume_solar", "volume_onshore_wind", "volume_offshore_wind"], axis=1, inplace=True)
        if mode == 'historical':
            df["fraction_renewable"] = df["volume_total_renewable"] / df["volume_total"]

        # add temperature data
        df_temp = DataManager.get_temperature(mode = mode,
                                              start_date = start_date,
                                              end_date = end_date)
        # the OpenMeteo API has more up-to-date data, and should always cover the whole NED data date range
        # so we always keep all the NED data
        df = df.merge(df_temp, left_index = True, right_on = 'date', how = 'inner')

        return df.set_index("date").rename_axis("ds")

    @staticmethod
    def get_temperature(mode: str = 'historical',
                        start_date: str = None,
                        end_date: str = None)  -> pd.DataFrame|None:
        """
        Retrieves air temperature data for the Netherlands from the Open-Meteo API.

        Args:
            mode: Either 'historical' or 'forecast'.
            start_date: Start date of time series in format YYYY-MM-DD.
            end_date: End date of time series in format YYYY-MM-DD.
        """
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        params = {
            "latitude": [52.107],
            "longitude": [5.179],
            "timezone": "UTC",
            "hourly": ["temperature_2m"],
            "models": "best_match",
            "start_date": start_date,
            "end_date": end_date
        }
        if mode == 'historical':
            url = "https://archive-api.open-meteo.com/v1/archive"


        elif mode == 'forecast':
            url = "https://api.open-meteo.com/v1/forecast"
        else:
            print(f"mode {mode} is not valid. Only 'historical' or 'forecast' accepted.")
            return

        responses = openmeteo.weather_api(url, params=params)
        hourly = responses[0].Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

        utc_datetimes = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )

        hourly_data = {"date": utc_datetimes}
        hourly_data["temperature_2m"] = hourly_temperature_2m

        return pd.DataFrame(data = hourly_data)

    def prepare_train_test_df(self, n_weeks: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        A wrapper function that generates the final training and test dataframes.
        Performs the following steps:
            1. Retrieves the most recent data n_weeks of data.
            2. Adds exogenous features
            3. Performs the train/test split

        Args:
            n_weeks: Length of retrieved data in number of weeks into the past.
                     The final week is used as test data.

        """
        df = self.get_last_n_weeks(n_weeks=n_weeks)
        # add extra day to account for API differences in start/ end date interpretation
        end_date = pd.Timestamp.now(tz='UTC')
        start_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(weeks=n_weeks) - pd.Timedelta(days=1)

        # Add exogenous features such as holiday, temperature etc.
        df = self.add_exo_features(df=df, mode='historical',
                                           start_date=start_date.strftime('%Y-%m-%d'),
                                           end_date=end_date.strftime('%Y-%m-%d'))

        # use the last week of data as final test data
        start_test_date = str(pd.Timestamp.now(tz='UTC') - pd.Timedelta(weeks=1))

        train_df, test_df = self.train_test_split(df=df, train_test_split_date=start_test_date,
                                                          test_end_date=str(pd.Timestamp.now(tz='UTC')))

        return train_df, test_df
