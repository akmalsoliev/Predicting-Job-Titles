import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from src.processing import DataPrep, Scaler, train_split
from src import model, KFold, RMSE, MSE


def main():
    #Removing tf-v1 errors
    tf.compat.v1.disable_eager_execution()

    ds = DataPrep("data/jobss.csv")
    #ds.get_df_report("preprocessed")
    ds()
    #ds.get_df_report("processed")
    ds.processed_export()
    df = ds.df

    target_col = "sal"
    optimizer = "adam"
    loss = "mse"
    model_metric = RMSE()
    kfold = KFold(n_folds=4, model=model, optimizer=optimizer, loss=loss, metrics=[model_metric])
    results = kfold(Scaler, df, target_col, batch_size=16, verbose=False)
    print("K-Fold mean result:", np.mean(results))

    #Tensorboard to debug
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./reports/tensorboard")

    ((train_data, train_labels), (test_data, test_labels)) = train_split(df, target=target_col)
    final_model = model()
    final_model.compile(optimizer=optimizer, loss=loss, metrics=[model_metric])
    final_model.fit(train_data, train_labels, epochs=100, batch_size=16, verbose=False, callbacks=[tensorboard])
    val_mse, val_rmse = final_model.evaluate(test_data, test_labels)
    final_model.save("models/final_model.h5")

    print("Final model's result: {}".format(val_rmse))


if __name__ == "__main__":
    main()
