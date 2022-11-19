from typing import Union
import numpy as np
import pandas as pd

def train_split(data:pd.DataFrame, target:str,test_size:float=.2) -> Union[list, tuple]:
    assert isinstance(data, pd.DataFrame), "DataFrame is required"
    train: pd.DataFrame = data.sample(frac=(1-test_size))
    test: pd.DataFrame = data.drop(index=train.index.tolist())
    
    def x_y_split(df:pd.DataFrame) -> Union[tuple, list]:
        return (df.drop(columns=target), df.loc(axis=1)[target])

    ((train_data, train_labels), (test_data, test_labels)) = \
            (x_y_split(train), x_y_split(test))

    return ((train_data, train_labels), (test_data, test_labels))
