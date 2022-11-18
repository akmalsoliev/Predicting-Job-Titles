import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras

from preprocessing import DataPrep

def main():
    ds = DataPrep("dataset/jobss.csv")
    ds()
    df = ds.df
    target_col = "sal"
    data, target = df.drop(columns=[target_col]), df[target_col]


if __name__ == "__main__":
    main()
