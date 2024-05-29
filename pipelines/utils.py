import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocess, DataSplit

def get_data_for_test():
    
    try:
        df = pd.read_csv("/home/amogh/College/MLOps/miniProject/data/olist_customers_dataset.csv")
        df = df.sample(n=100)
        preprocess = DataPreprocess()
        data_cleaning = DataCleaning(df, preprocess)
        df = data_cleaning.handle_data()
        df.drop(['review_score'],axis=1, inplace=True)
        result = df.to_json(orient='split')
        return result
    
    except Exception as e:
        logging.error(e)
        raise e
        
