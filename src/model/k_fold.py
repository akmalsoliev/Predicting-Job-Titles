from typing import Callable, Union
from numpy._typing import NDArray
import pandas as pd
import numpy as np


class KFold:
    def __init__(self, n_folds:int):
        self.n_folds = n_folds

    def __data_label_split__(self, data:NDArray):
        return np.delete(data, self.target_index, axis=1), data[:, self.target_index]

    def call(self, data:Union[pd.DataFrame, NDArray], Scaler, 
             model:Callable, target:Union[str, int]):

        if isinstance(target, str) and isinstance(data, pd.DataFrame):
            self.target_index: int = np.where(data.columns==target)[0][0]
        elif isinstance(target, int) and isinstance(data, NDArray):
            self.target_index: int = target
        else:
            raise TypeError("Miss-match in data and target type.")

        split_data = np.array_split(data, self.n_folds)
        metrics_list: list[float] = []

        for fold in range(self.n_folds):
            train_index = [index for index in range(self.n_folds) if index =! fold]
            train_ds = np.concatenate(split_data[train_index], axis=0)
            test_ds = split_data[fold]
            
            (train_data, train_labels) = self.__data_label_split__(train_ds)
            (test_data, test_labels) = self.__data_label_split__(test_ds) 

            # This is an extremely lazy of implementing this, just applying scaling 
            # on the hole dataset, not recommended, but it won't effect my data as
            # half of it is hot-encoded.
            standard_scaler = Scaler(train_data, test_data)
            train_data = standard_scaler.train_scale()
            test_data = standard_scaler.test_scale()

            model = model()

            # will finish tomorrow, need to build a model then append metrics to a list 
            # and see the results with KFold.
