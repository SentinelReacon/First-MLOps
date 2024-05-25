import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from.config import ModelNameConfig

@step
def training(
    xtrain: pd.DataFrame,
    xtest: pd.DataFrame,
    ytrain: pd.Series,
    ytest: pd.Series, 
    config: ModelNameConfig
) -> RegressorMixin:
    
    
    model = None
    
    try:
        if config.model_name == "Linear Regression":
            model - LinearRegressionModel()
            trained_model = model.train(xtrain, ytrain)
            return trained_model
        
        else: 
            raise ValueError("Model not supported")
            
    except Exception as e:
        logging.error(f"Model training not supported {e}")