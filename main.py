import numpy as np
import pandas as pd 
# import tensorflow as tf
# from tensorflow import keras
from src.processing import DataPrep, train_split, Scaler


def main():
    ds = DataPrep("data/jobss.csv")
    ds.get_df_report("preprocessed")
    ds()
    ds.get_df_report("processed")
    ds.processed_export()
    df = ds.df

    target_col = "sal"
    (train_data, train_labels), (test_data, test_labels) = train_split(df, target_col)

    scale_cols = ["Longitude", "Latitude"]
    standard_scaler = Scaler(train_data[scale_cols], test_data[scale_cols])
    train_data[scale_cols] = standard_scaler.train_scale()
    test_data[scale_cols] = standard_scaler.test_scale()

if __name__ == "__main__":
    main()
