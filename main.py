import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from src.processing import DataPrep, Scaler
from src import model, KFold


def main():
    ds = DataPrep("data/jobss.csv")
    #ds.get_df_report("preprocessed")
    ds()
    #ds.get_df_report("processed")
    ds.processed_export()
    df = ds.df

    target_col = "sal"
    kfold = KFold(n_folds=4, model=model, optimizer="adam", loss="mae", metrics=["mae"])
    results = kfold(Scaler, df, target_col, verbose=False)
    print("The mean is follow:", df[target_col].mean())
    print("The std is follow:", df[target_col].std())
    print(results)


if __name__ == "__main__":
    main()
