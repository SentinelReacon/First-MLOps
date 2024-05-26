import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate(
    model: RegressorMixin,
    xtest: pd.DataFrame,
    ytest: pd.Series
) -> Annotated[float, "mse_score"]:
    
    try:
        pred = model.predict(xtest)
        mse = MSE()
        mse_score = mse.calculate_score(ytest, pred)
        mlflow.log_metric("Mean Squared Error score", mse_score)
        
        return mse_score
    
    except Exception as e:
        logging.error(f"Error in calculating loss {e}")
        raise e