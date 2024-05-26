import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataSplit, DataPreprocess
from typing_extensions import Annotated
from typing import Tuple

@step
def cleaning(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "xtrain"],
    Annotated[pd.DataFrame, 'xtest'],
    Annotated[pd.Series, 'ytrain'],
    Annotated[pd.Series, 'ytest']
]:
    
    """
    This particular steps takes dataframe as an input and returns xtrain, xtest, ytrain and ytest.
    Applies all the preprocessing and the splitting functions from src.
    """
    
    
    try:
        
        preprocess = DataPreprocess()
        data_cleaning = DataCleaning(df, preprocess)
        processed_data = data_cleaning.handle_data()
        divide = DataSplit()
        data_splitting = DataCleaning(processed_data, divide)
        xtrain, xtest, ytrain, ytest = data_splitting.handle_data()
        logging.info("Data Cleaning and Splittin completed")
        
        return xtrain, xtest, ytrain, ytest
        
    except Exception as e:
        logging.error(f"Error in Data pre processing and cleaning {e}")
        raise e