from typing import Callable, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy._typing import NDArray

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

    def __call__(self, scaler:Scaler, data:Union[pd.DataFrame, NDArray], target:Union[str, int], epochs:int=100, **kwargs):
        if isinstance(target, str) and isinstance(data, pd.DataFrame):
            self.target_index: int = np.where(data.columns==target)[0][0]
        elif isinstance(target, int) and isinstance(data, NDArray):
            self.target_index: int = target
        else:
            raise TypeError("Miss-match in data and target type.")

        split_data = np.array_split(data.to_numpy(), self.n_folds)
        split_array: NDArray = np.array(split_data)
        metrics_list: list[float] = []

        for fold in range(self.n_folds):
            train_index = [index for index in range(self.n_folds) if index != fold]
            train_ds: NDArray = np.vstack(split_array[train_index])
            test_ds: NDArray = split_array[fold]
            
            (train_data, train_labels) = self.__data_label_split__(train_ds)
            (test_data, test_labels) = self.__data_label_split__(test_ds) 

            # This is an extremely lazy of implementing scaling, just applying scaling 
            # on the hole dataset, not recommended, but it won't affect my data as
            # except lat and long is hot-encoded.
            standard_scaler = scaler(train_data, test_data)
            train_data = standard_scaler.train_scale()
            test_data = standard_scaler.test_scale()

            self.model.fit(train_data, train_labels, epochs=epochs, **kwargs)

            plt.plot(np.arange(len(test_labels)), test_labels, label="test")
            plt.plot(np.arange(len(test_labels)), self.model.predict(test_data), label="prediction")
            plt.legend()
            plt.show()

            model_mse, model_mae = self.model.evaluate(test_data, test_labels)
            metrics_list.append(model_mse)
        
        return metrics_list
