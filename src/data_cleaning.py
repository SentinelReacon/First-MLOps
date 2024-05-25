import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union

from sklearn.model_selection import train_test_split

class Data(ABC):
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    

class DataPreprocess(Data):
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
    
        try:
            data = data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp"
            ], axis=1)
            
            data['product_weight_g'].fillna(data['product_weight_g'].mean(), inplace=True)
            data['product_length_cm'].fillna(data['product_length_cm'].mean(), inplace=True)
            data['product_height_cm'].fillna(data['product_height_cm'].mean(), inplace=True)
            data['product_width_cm'].fillna(data['product_width_cm'].mean(), inplace=True)
            data['review_comment_message'].fillna("No Review", inplace=True)
            
            data = data.select_dtypes([np.number])
            cols_to_drop = ['customer_zip_code_prefix', 'order_item_id']
            data = data.drop(cols_to_drop, axis=1)
            
            return data
            
            
        except Exception as e:
            logging.error(f"Error in processing data {e}")
            raise e
        
        
class DataSplit(Data):
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        
        try:
            X = data.drop(['review_score'], axis=1)
            y = data['review_score']
            xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, shuffle=True)
            
            return xtrain, xtest, ytrain, ytest
        
        except Exception as e:
            logging.error("Error in splitting the data {}".format(e))
            raise e
        
        
class DataCleaning:
    
    """
    this is basically the class which implements any of the above strategies (or classes).
    what u gotta do is instead of calling any of the classes above, just call the class below and simply
    pass the name of the above class which u want to inherit.   
    """
    
    def __init__(self, data: pd.DataFrame, strategy: Data):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data {e}")
            raise e
