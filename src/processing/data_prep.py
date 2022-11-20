from re import T
from typing import List, Optional
import numpy as np 
import pandas as pd 

from src.util import eda_report
from src.processing import OneHotEncode

class DataPrep:
    def __init__(self, dataset:str):
        self.ds_path = dataset
        try: 
            self.df = self.__get_df__()
        except pd.errors.EmptyDataError:
            print("""File is not csv, setting `self.df` to None, modify using setter""")
            self.df = None

    def __get_df__(self) -> pd.DataFrame:
        return pd.read_csv(self.ds_path)

    def __sequential_construct__(self, remove_cols:List[str]=["Job Title"]):
        assert isinstance(self.df, pd.DataFrame), "`self.df` needs to be a DataFrame!"
        "Usually nan column is named `Unnamed`"
        drop_cols = [col for col in self.df.columns if "Unnamed" in col]
        self.df.drop(columns=drop_cols, inplace=True)

        """Removing Cols
        Removing "Job Title" too many unique values present 
        in the column, possible effect on the outcome 
        is very low."""
        self.df.drop(columns=remove_cols, inplace=True)

        '''There are values named "vide", they'll be replaced with "missing"'''
        self.df.replace("vide", "missing", inplace=True)

        
        "Fill Missing" 
        fill_col:pd.Index = self.df.select_dtypes(include=[object]).columns
        self.df.loc(axis=1)[fill_col] = self.df.loc(axis=1)[fill_col].fillna("missing")

        "Drop NA" 
        self.df = self.df.dropna()

        "OHE Cols"
        def multi_col_ohe(df:pd.DataFrame, col:str, sep:str="|") -> pd.DataFrame:
            ohe_key =  OneHotEncode(df, col)
            ohe_key.text_split(sep=sep)
            ohe_key.top_x_vals(10)
            ohe_key.encode(inplace=True)
            return ohe_key.df

        self.df = multi_col_ohe(self.df, col="Key Skills")
        self.df = multi_col_ohe(self.df, col="Functional Area", sep=",")
        self.df = multi_col_ohe(self.df, col="Industry", sep=",")


        def single_col_ohe(df:pd.DataFrame, col:str) -> pd.DataFrame:
            keep_rows = df[col].value_counts().head(10).index
            df.loc[df[col].isin(keep_rows) == False, col] = np.nan
            ohe_jer = OneHotEncode(df, col)
            ohe_jer.encode(inplace=True)
            return ohe_jer.df

        single_ohe_cols = ["Role Category", "Location", "Role", "Job Experience Required"]

        for col in single_ohe_cols:
            self.df = single_col_ohe(self.df, col)
        
    def __call__(self):
        self.__sequential_construct__()

    def processed_export(self):
        assert isinstance(self.df, pd.DataFrame), "DataFrame needs to exist to export it"
        self.df.to_csv("{}-processed.csv".format(self.ds_path.replace("csv", "")))

        
    def get_df_report(self, file_name:str="reports/report.html"):
        if file_name:
            file_name="reports/{}_report.html".format(file_name)
        eda_report(self.df, file_name)
