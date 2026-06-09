from src.data_manager import DataManager
from src.model_trainer import ModelTrainer, ModelConfig
from src.utils import slice_last_n_weeks, convert_to_ag_df
import tempfile
import pandas as pd
from pathlib import Path

# SETTINGS
N_WEEKS = 27
WINDOW_WEEKS = [4,8,12,26]
SAVE_DATA_PATH = Path("artifacts/data/train-test/")
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

data_manager = DataManager()
df = data_manager.get_last_n_weeks(n_weeks = N_WEEKS)
# add extra day to account for API differences
end_date = pd.Timestamp.now(tz='UTC')
start_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(weeks = N_WEEKS)  - pd.Timedelta(days=1)

df = data_manager.add_exo_features(df = df, mode = 'historical',
                                   start_date = start_date.strftime('%Y-%m-%d'),
                                   end_date = end_date.strftime('%Y-%m-%d'))

# use the last week of data as final test data
start_test_date = str(pd.Timestamp.now(tz='UTC') - pd.Timedelta(weeks = 1) )

train_df, test_df = data_manager.train_test_split(df = df, train_test_split_date = start_test_date,
                                                  test_end_date = str(pd.Timestamp.now(tz='UTC')) )

with tempfile.TemporaryDirectory() as tmp:
    config = ModelConfig(path=tmp)
    trainer = ModelTrainer(config)

    best_train_window = trainer.find_optimal_training_window(train_df = train_df, window_weeks = WINDOW_WEEKS)

# train model with the best training window
train_df_opt = slice_last_n_weeks(df = train_df, n_weeks = best_train_window)
train_df_opt = convert_to_ag_df(train_df_opt)

config = ModelConfig(path='artifacts/models/direct')
trainer = ModelTrainer(config)
predictor = trainer.fit_predictor(train_df = train_df_opt)

# save the train and test data
train_df_opt.to_csv(SAVE_DATA_PATH / "train_direct.csv")
test_df.to_csv(SAVE_DATA_PATH / "test_direct.csv")