from autogluon.timeseries import TimeSeriesPredictor
from src.utils import convert_to_ag_df, slice_last_n_weeks
from src.data_manager import DataManager
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
import json
import tempfile

@dataclass
class ModelConfig:
    horizon: int = 24 * 7
    frequency: str = 'h'
    target: str = "emissionfactor"
    exo_features: tuple = ("volume_total_renewable", "is_holiday", "temperature_2m")
    eval_metric: str = "mae"
    num_val_windows: int = 2
    path: str | None = None # SAVE path for the trained model
    hyperparams: dict = field(default_factory=lambda: {
        "Chronos2": [{"fine_tune": False}],
        "SeasonalNaive": {},
        "RecursiveTabular": {},
        "DirectTabular": {},
        "DynamicOptimizedTheta": {},
    })

class ModelTrainer:
    """Methods for all things model training."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        if self.config.path:
            Path(self.config.path).mkdir(parents=True, exist_ok=True)


    def fit_predictor(self, train_df: pd.DataFrame = None) -> TimeSeriesPredictor:
        """
        Fit the predictor using the parameters specified at initialisation of the model trainer.
        """
        predictor = TimeSeriesPredictor(
            prediction_length = self.config.horizon,
            freq = self.config.frequency,
            target = self.config.target,
            known_covariates_names = list(self.config.exo_features),
            eval_metric = self.config.eval_metric,
            path = self.config.path
        )

        predictor.fit(
            train_df,
            num_val_windows=self.config.num_val_windows,
            hyperparameters = self.config.hyperparams,

        )

        return predictor

    def find_optimal_training_window(self, train_df: pd.DataFrame = None,
                                     window_weeks: list[int] = None,
                                     time_col: str = "ds"):
        """
        Compares user defined training windows (i.e. the length of the training data), returning the model with the best evaluation score.
        """

        results = {}

        for weeks in window_weeks:
            subset = slice_last_n_weeks(df = train_df, n_weeks=weeks, time_col=time_col)
            subset = convert_to_ag_df(subset, time_col= time_col)

            predictor = self.fit_predictor(train_df=subset)

            best_score = predictor.info()['best_model_score_val']
            results[f"week_{weeks}"] = best_score

        # get the max MAE, since autogluon returns errors as negatives (i.e., "higher is better")
        return int( max(results, key=results.get)[5:] )

    def train_direct_model(self, n_weeks: int = None, window_weeks: list[int] = None,
                           save_data_path:str = None, save_model_path:str = None,
                           best_train_window: int = None) -> None:
        """
        Trains the direct model. Saves the train/test dataset to save_data_path.

        Args:
            n_weeks: Length of retrieved training data in number of weeks into the past.
            window_weeks: List of potential dataset lengths in weeks, used to find the optimal training window.
                          Only required if "best_train_window" is None.
            best_train_window: Sets the optimal training window (skips searching through "window_weeks" list).
            save_data_path: Path to save the train and test data (validation data is handled in the ModelConfig).
            save_model_path: Path to save the model artifacts.

        """

        SAVE_DATA_PATH = Path(save_data_path)
        SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

        data_manager = DataManager()
        df = data_manager.get_last_n_weeks(n_weeks=n_weeks)
        # add extra day to account for API differences in start/ end date interpretation
        end_date = pd.Timestamp.now(tz='UTC')
        start_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(weeks=n_weeks) - pd.Timedelta(days=1)

        # Add exogenous features such as holiday, temperature etc.
        df = data_manager.add_exo_features(df=df, mode='historical',
                                           start_date=start_date.strftime('%Y-%m-%d'),
                                           end_date=end_date.strftime('%Y-%m-%d'))

        # use the last week of data as final test data
        start_test_date = str(pd.Timestamp.now(tz='UTC') - pd.Timedelta(weeks=1))

        train_df, test_df = data_manager.train_test_split(df=df, train_test_split_date=start_test_date,
                                                          test_end_date=str(pd.Timestamp.now(tz='UTC')))

        if best_train_window is None:
            with tempfile.TemporaryDirectory() as tmp:
                config = ModelConfig(path=tmp)
                trainer = ModelTrainer(config)

                best_train_window = trainer.find_optimal_training_window(train_df=train_df, window_weeks=window_weeks)

        # train model with the best training window
        train_df_opt = slice_last_n_weeks(df=train_df, n_weeks=best_train_window)
        train_df_opt = convert_to_ag_df(train_df_opt)

        config = ModelConfig(path=save_model_path)
        trainer = ModelTrainer(config)
        predictor = trainer.fit_predictor(train_df=train_df_opt)

        with open(Path(config.path) / "train_config.json", "w") as f:
            json.dump({"best_train_window": best_train_window}, f)

        # save the train and test data
        train_df_opt.to_csv(SAVE_DATA_PATH / "train_direct.csv")
        test_df.to_csv(SAVE_DATA_PATH / "test_direct.csv")
