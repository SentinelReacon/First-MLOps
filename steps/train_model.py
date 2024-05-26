import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from.config import ModelNameConfig

@step
def training(
    xtrain: pd.DataFrame,
    ytrain: pd.DataFrame,
    xtest: pd.DataFrame,
    ytest: pd.DataFrame,
    config: ModelNameConfig
) -> RegressorMixin:
    
    
    try:

        model = LinearRegressionModel()
        trained_model = model.train(xtrain, ytrain)
        return trained_model

            
    except Exception as e:
        logging.error(f"Erorr in model training {e}")