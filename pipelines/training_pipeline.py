import logging
from zenml import pipeline
import pandas as pd
from steps.ingest_data import ingest_data
from steps.clean_data import cleaning
from steps.train_model import training
from steps.eval_model import evaluate

"""
this pipeline decorator uses cached version of the data every time a new run is there if there is no change in 
the data.
""" 
@pipeline(enable_cache=True)
def train_pipe(
    data_path: str
):
    df = ingest_data(data_path)
    xtrain, xtest, ytrain, ytest = cleaning(df)
    model_lr = training(xtrain, ytrain, xtest, ytest)
    mse_lr = evaluate(model_lr, xtest, ytest)