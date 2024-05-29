import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from.config import ModelNameConfig

import mlflow
from zenml.client import Client

"""
this experiment tracker is something which tracks all the runs
"""
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def training(
    xtrain: pd.DataFrame,
    ytrain: pd.DataFrame,
    xtest: pd.DataFrame,
    ytest: pd.DataFrame,
    config: ModelNameConfig
) -> RegressorMixin:
    
    
    try:
        """
        auto logs all the model scores and all its parameters (information about the model)
        """
        mlflow.sklearn.autolog()
        model = LinearRegressionModel()
        trained_model = model.train(xtrain, ytrain)
        return trained_model

            
    except Exception as e:
        logging.error(f"Error in model training {e}")