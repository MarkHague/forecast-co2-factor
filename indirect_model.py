from src.data_manager import DataManager
from src.model_trainer import ModelTrainer, ModelConfig
from src.utils import slice_last_n_weeks, convert_to_ag_df
import tempfile
import json
import pandas as pd
from pathlib import Path


# ---------------------- MODEL 1 - PREDICT TOTAL VOLUME ------------------ #
# SETTINGS
N_WEEKS = 27
WINDOW_WEEKS = [4,8,12,26]
SAVE_DATA_PATH = Path("artifacts/data/train-test/")
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

data_manager = DataManager()
df = data_manager.get_last_n_weeks(n_weeks = N_WEEKS)
df = data_manager.add_exo_features(df = df)

# use the last week of data as final test data
# also reserve a week for validation data needed for this 2-step prediction
start_test_date = str(pd.Timestamp.now(tz='UTC') - pd.Timedelta(weeks = 2) )

train_df, test_df = data_manager.train_test_split(df = df, train_test_split_date = start_test_date,
                                                  test_end_date = str(pd.Timestamp.now(tz='UTC')) )

# predict total volume, not emission factor
with tempfile.TemporaryDirectory() as tmp:
    config = ModelConfig(path=tmp, target = "volume_total")
    trainer = ModelTrainer(config)

    best_train_window = trainer.find_optimal_training_window(train_df = train_df, window_weeks = WINDOW_WEEKS)

# train model with the best training window
train_df_opt = slice_last_n_weeks(df = train_df, n_weeks = best_train_window)
train_df_opt = convert_to_ag_df(train_df_opt)

config = ModelConfig(path='artifacts/models/indirect', target = "volume_total")
trainer = ModelTrainer(config)
predictor = trainer.fit_predictor(train_df = train_df_opt)

with open(Path(config.path) / "train_config.json", "w") as f:
    json.dump({"best_train_window": best_train_window}, f)

# ---------------------- MODEL 2 - PREDICT EMISSION FACTOR ------------------ #
# use a simple OLS, since CO2 factor should linearly depend on fraction of renewables
import statsmodels.api as sm
from statsmodels.tools.eval_measures import meanabs

X = sm.add_constant(train_df["fraction_renewable"].values)
y = train_df["emissionfactor"].values

model = sm.OLS(y, X)
results = model.fit()

# ---------------------- COMBINE MODELS, GET FINAL VALIDATION SCORE ------------------ #
val_df = test_df.head(24*7) # use week 1 of test data
val_df = convert_to_ag_df(val_df)
train_df = convert_to_ag_df(train_df)

# step 1 - predict the total volume
preds_vol_total = predictor.predict(data = train_df,
                                   known_covariates = val_df.drop(["volume_total"], axis = 1) )

fraction_renewables_pred = sm.add_constant(val_df["volume_total_renewable"].values / preds_vol_total["mean"].values )

preds_emission_factor = results.predict(fraction_renewables_pred)
mae = meanabs(preds_emission_factor, val_df["emissionfactor"])

print("\n")
print("-------------------------------------------------------------------------")
print(f"Total MAE: {mae}")