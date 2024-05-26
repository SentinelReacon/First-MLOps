import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error

class EvaluateModel(ABC):
    
    """
    this class is for evaluating the models.
    """
    
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        pass
    
    
class MSE(EvaluateModel):
    
    """
    cost function -> MSE
    """
    
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try:
            logging.info("Calculating the cost value")
            loss = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {loss}")
            return loss
        except Exception as e:
            logging.error(f"Error in calculating MSE {e}")            