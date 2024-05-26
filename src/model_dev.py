import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    
    """
    this is going to be the abstract class for all the models
    """
    
    @abstractmethod
    def train(xtrain, ytrain):
        
        pass
    

class LinearRegressionModel(Model):
    
    def train(self, xtrain, ytrain, **kwargs):
        
        try:
            lr = LinearRegression(**kwargs)
            lr.fit(xtrain, ytrain)
            logging.info("Model training completed")
            
            return lr
        except Exception as e:
            logging.error(f"Error in training the model {e}")
            raise e
        