from typing import Union
import os 
import pandas as pd

class DataSet:
    def __init__(self, ds:str):
        self.ds = ds
        self.df:pd.DataFrame = self.__load_ds__(self.ds)
    def __load_ds__(self, ds)->pd.DataFrame:
        return pd.read_csv(ds)
    def drop_cols(self):
        drop_cols = self.df.columns.str.contains("Unnamed")
        self.df.drop(columns=drop_cols, inplace=True)
    def extract(self):
        self.df.to_parquet(self.ds.replace("csv", "parque"))


