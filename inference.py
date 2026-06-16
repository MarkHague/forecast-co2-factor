import json
import pandas as pd

from src.data_manager import DataManager
from autogluon.timeseries import TimeSeriesPredictor

from src.utils import convert_to_ag_df

# get NED forecast for renewables
data_manager = DataManager()
now = pd.Timestamp.now(tz='UTC')
end_forecast = now + pd.Timedelta(days = 7)

forecast_data = data_manager.get_forecast_data()

# add exogenous data
# the openmeteo API starts the forecast from the current day, but NED goes from 11pm of previous day
start_temperature = now - pd.Timedelta(days = 1)
end_temperature = now + pd.Timedelta(days = 7)
df = data_manager.add_exo_features(df = forecast_data, mode = 'forecast',
                                   start_date = start_temperature.strftime('%Y-%m-%d'),
                                   end_date = end_temperature.strftime('%Y-%m-%d'))

# load the desired model and its training config
predictor = TimeSeriesPredictor.load("artifacts/models/direct")

with open("artifacts/models/direct/train_config.json") as f:
    train_config = json.load(f)

# fetch historical context needed for prediction
historical_data = data_manager.get_last_n_weeks(n_weeks=train_config["best_train_window"])
# add extra day to account for API differences
end_date = pd.Timestamp.now(tz='UTC')
start_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(weeks = train_config["best_train_window"])  - pd.Timedelta(days=1)

historical_data = data_manager.add_exo_features(df = historical_data, mode = 'historical',
                                   start_date = start_date.strftime('%Y-%m-%d'),
                                   end_date = end_date.strftime('%Y-%m-%d'))

historical_data = convert_to_ag_df(historical_data)

df_forecast = convert_to_ag_df(df)

preds = predictor.predict(data = historical_data, known_covariates = df_forecast)
preds.to_csv("artifacts/data/predictions/latest_preds.csv")