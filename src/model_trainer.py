from autogluon.timeseries import TimeSeriesPredictor
from src.utils import convert_to_ag_df, slice_last_n_weeks
from src.data_manager import DataManager
import statsmodels.api as sm
from statsmodels.tools.eval_measures import meanabs
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
            
            # get the best score using the metric of choice
            best_score = predictor.info()['best_model_score_val']
            results[f"week_{weeks}"] = best_score

        # get the max MAE, since autogluon returns errors as negatives (i.e., "higher is better")
        return int( max(results, key=results.get)[5:] )

    def train_direct_model(self, n_weeks: int = None, window_weeks: list[int] = None,
                           save_data_path: str = None,
                           best_train_window: int = None) -> None:
        """
        Trains the direct model. Saves the train/test dataset to save_data_path.
        Note that the last week of data is reserved for test data, reducing the training data by 1 week. For example, if n_weeks = 4, then only 3 weeks will be used to train the model (even if best_train_window = 4). 

        Args:
            n_weeks: Length of retrieved training data in number of weeks into the past.
            window_weeks: List of potential dataset lengths in weeks, used to find the optimal training window.
                          Only required if "best_train_window" is None. max(window_weeks) must be less than or equal to n_weeks.
            best_train_window: Sets the optimal training window (skips searching through "window_weeks" list).
            save_data_path: Path to save the train and test data (validation data is handled in the ModelConfig).

        """

        save_path = Path(save_data_path)
        save_path.mkdir(parents=True, exist_ok=True)

        data_manager = DataManager()
        train_df, test_df = data_manager.prepare_train_test_df(n_weeks=n_weeks)

        if best_train_window is None:
            with tempfile.TemporaryDirectory() as tmp:
                config = ModelConfig(path=tmp)
                trainer = ModelTrainer(config)

                best_train_window = trainer.find_optimal_training_window(train_df=train_df, window_weeks=window_weeks)

        # train model with the best training window
        train_df_opt = slice_last_n_weeks(df=train_df, n_weeks=best_train_window)
        train_df_opt = convert_to_ag_df(train_df_opt)

        predictor = self.fit_predictor(train_df=train_df_opt)

        with open(Path(self.config.path) / "train_config.json", "w") as f:
            json.dump({"best_train_window": best_train_window}, f)

        # save the train and test data
        train_df_opt.to_csv(save_path / "train_direct.csv")
        test_df.to_csv(save_path / "test_direct.csv")

    def train_indirect_model(self, n_weeks: int = None, window_weeks: list[int] = None,
                             save_data_path: str = None,
                             best_train_window: int = None) -> float:
        """
        Trains the indirect model (two-step: volume_total via AutoGluon + emissionfactor via OLS).
        Saves the AutoGluon predictor and OLS model to self.config.path, and train/test data to save_data_path.
        Note that the last 2 weeks of data are reserved for test data, reducing the training data by 2 weeks. For example, if n_weeks = 4, then only 2 weeks will be used to train the model (even if best_train_window = 4). 

        Args:
            n_weeks: Length of retrieved training data in number of weeks into the past.
            window_weeks: List of potential dataset lengths in weeks, used to find the optimal training window.
                          Only required if "best_train_window" is None. max(window_weeks) must be <= n_weeks.
            best_train_window: Sets the optimal training window (skips searching through "window_weeks" list).
            save_data_path: Path to save the train and test data.
        Returns:
            Total MAE for the 2-step indirect model. 
        """
        self.config.target = "volume_total"

        save_path = Path(save_data_path)
        save_path.mkdir(parents=True, exist_ok=True)

        data_manager = DataManager()
        # use the last week of data as final test data
        # also reserve a week for validation data needed for this 2-step prediction
        train_df, test_df = data_manager.prepare_train_test_df(n_weeks=n_weeks, n_test_weeks=2)

        if best_train_window is None:
            with tempfile.TemporaryDirectory() as tmp:
                config = ModelConfig(path=tmp, target=self.config.target)
                trainer = ModelTrainer(config)
                best_train_window = trainer.find_optimal_training_window(train_df=train_df, window_weeks=window_weeks)

        train_df_opt = slice_last_n_weeks(df=train_df, n_weeks=best_train_window)
        train_df_opt = convert_to_ag_df(train_df_opt)

        predictor = self.fit_predictor(train_df=train_df_opt)

        with open(Path(self.config.path) / "train_config.json", "w") as f:
            json.dump({"best_train_window": best_train_window}, f)

        # OLS: fraction_renewable -> emissionfactor
        X = sm.add_constant(train_df["fraction_renewable"].values)
        y = train_df["emissionfactor"].values
        ols_results = sm.OLS(y, X).fit()
        ols_results.save(str(Path(self.config.path) / "ols_model.pkl"))

        # combined validation on week 1 of test data
        val_df = convert_to_ag_df(test_df.head(24 * 7))
        train_df_ag = convert_to_ag_df(train_df)

        preds_vol_total = predictor.predict(
            data=train_df_ag,
            known_covariates=val_df.drop(["volume_total"], axis=1)
        )
        fraction_renewables_pred = sm.add_constant(
            val_df["volume_total_renewable"].values / preds_vol_total["mean"].values
        )
        preds_emission_factor = ols_results.predict(fraction_renewables_pred)
        mae = meanabs(preds_emission_factor, val_df["emissionfactor"])

        train_df_opt.to_csv(save_path / "train_indirect.csv")
        test_df.to_csv(save_path / "test_indirect.csv")
        
        return mae
