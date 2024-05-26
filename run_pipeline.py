from pipelines.training_pipeline import train_pipe
from zenml.client import Client


if __name__ == "__main__":
    
    """
    so in this Tracking URI is the information of the run. Just copy the URI and run the command 

                mlflow ui --backend-store-uri <paste the uri here>
                
    this will create a mlflow UI and there you can see all the logs such as the metrics and dataset and others.
    the tracking uri will be printed in the first line itself. 
    
    """
    
    print(f"Tracking URI: {Client().active_stack.experiment_tracker.get_tracking_uri()}")
    train_pipe(data_path = '/home/amogh/College/MLOps/miniProject/data/olist_customers_dataset.csv')