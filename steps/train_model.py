import logging
import pandas as pd
from zenml import step

@step
def training(
    df: pd.DataFrame
) -> None:
    pass