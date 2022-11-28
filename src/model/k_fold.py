from typing import Callable, Union, Type
from numpy._typing import NDArray
import pandas as pd
import numpy as np
from ..processing import Scaler


class KFold:
    def __init__(self, n_folds:int, model:Callable, optimizer:Union[str, Callable], loss:Union[str, Callable], metrics:list):
        self.n_folds = n_folds
        self.model = model()
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.compiled_model = self._compile_model_()

    def __data_label_split__(self, data:NDArray):
        return np.delete(data, self.target_index, axis=1), data[:, self.target_index]

    def _compile_model_(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def call(self, scaler:Scaler, data:Union[pd.DataFrame, NDArray], target:Union[str, int], epochs:int=100, **kwargs):

        if isinstance(target, str) and isinstance(data, pd.DataFrame):
            self.target_index: int = np.where(data.columns==target)[0][0]
        elif isinstance(target, int) and isinstance(data, NDArray):
            self.target_index: int = target
        else:
            raise TypeError("Miss-match in data and target type.")

        split_data = np.array_split(data, self.n_folds)
        metrics_list: list[float] = []

        for fold in range(self.n_folds):
            train_index = [index for index in range(self.n_folds) if index != fold]
            train_ds = np.concatenate(split_data[train_index], axis=0)
            test_ds = split_data[fold]
            
            (train_data, train_labels) = self.__data_label_split__(train_ds)
            (test_data, test_labels) = self.__data_label_split__(test_ds) 

            # This is an extremely lazy of implementing scaling, just applying scaling 
            # on the hole dataset, not recommended, but it won't effect my data as
            # except lat and long is hot-encoded.
            standard_scaler = scaler(train_data, test_data)
            train_data = standard_scaler.train_scale()
            test_data = standard_scaler.test_scale()

            self.model.fit(train_data, train_labels, epochs=epochs, kwargs=kwargs)

            metrics_list.append(self.model.evaluate(test_data, test_labels))
