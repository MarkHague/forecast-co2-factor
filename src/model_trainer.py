from autogluon.timeseries import TimeSeriesPredictor
from src.utils import convert_to_ag_df, slice_last_n_weeks
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

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

