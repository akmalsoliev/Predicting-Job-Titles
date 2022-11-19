import numpy as np
import pandas as pd


class Scaler:
    def __init__(self, train_data, test_data) -> None:
        self.train_data = train_data
        self.test_data = test_data

    def __standard(self):
        self.mean = np.mean(self.train_data, axis=1)
        self.std = np.std(self.train_data, axis=1)

    def train_scale(self):
        return (self.train_data - self.mean) / self.std

    def test_scale(self):
        return (self.test_data - self.mean) / self.std

    def apply(self, data):
        return (data - self.mean) / self.std
