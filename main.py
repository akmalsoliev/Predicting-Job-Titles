import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras

from preprocessing import DataPrep, train_split, Scaler

def main():
    ds = DataPrep("dataset/jobss.csv")
    ds()
    df = ds.df
    target_col = "sal"

    (train_data, train_labels), (test_data, test_labels) = train_split(df, target_col)

    standard_scaler = Scaler(train_data, test_data)
    train_data = standard_scaler.train_data()
    test_data = standard_scaler.test_data()


if __name__ == "__main__":
    main()
