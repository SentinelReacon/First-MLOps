import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated

@step
def evaluate(
    model: RegressorMixin,
    xtest: pd.DataFrame,
    ytest: pd.Series
) -> Annotated[float, "mse_score"]:
    
    try:
        pred = model.predict(xtest)
        mse = MSE()
        mse_score = mse.calculate_score(ytest, pred)
        
        return mse_score
    
    except Exception as e:
        logging.error(f"Error in calculating loss {e}")
        raise e