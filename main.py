from typing import List
import numpy as np 
import pandas as pd 

from util import DataSet, eda_report
from preprocessing import OneHotEncode

def main():
    ds = DataSet("dataset/jobss.csv")
    ds.drop_cols()
    df = ds.df
    
    ##### Removing Cols ##### 
    # Removing "Job Title" too many unique values present 
    # in the column, possible effect on the outcome 
    # is very low.
    remove_cols: List[str]= ["Job Title"]
    df.drop(columns=remove_cols, inplace=True)

    ##### Rename Val ##### 
    df.replace("vide", "missing", inplace=True)
    
    ##### Fill Missing ##### 
    fill_col:pd.Index = df.select_dtypes(include=[object]).columns
    df.loc(axis=1)[fill_col] = df.loc(axis=1)[fill_col].fillna("missing")
    
    ##### Drop NA ##### 
    df = df.dropna()

    # OHE Cols: "Job Experience Required", "Key Skills",
    # "Role Category", 
    def multi_col_ohe(df:pd.DataFrame, col:str, sep:str="|") -> pd.DataFrame:
        ohe_key =  OneHotEncode(df, col)
        ohe_key.text_split(sep=sep)
        ohe_key.top_x_vals(10)
        ohe_key.encode(inplace=True)
        return ohe_key.df

    df = multi_col_ohe(df, col="Key Skills")
    df = multi_col_ohe(df, col="Functional Area", sep=",")
    df = multi_col_ohe(df, col="Industry", sep=",")


    def single_col_ohe(df:pd.DataFrame, col:str) -> pd.DataFrame:
        keep_rows = df[col].value_counts().head(10).index
        df.loc[df[col].isin(keep_rows) == False, col] = np.nan
        ohe_jer = OneHotEncode(df, col)
        ohe_jer.encode(inplace=True)
        return ohe_jer.df

    single_ohe_cols = ["Role Category", "Location", "Role", "Job Experience Required"]

    for col in single_ohe_cols:
        df = single_col_ohe(df, col)

    df.to_csv("dataset/test.csv")
    df.cov().to_csv("dataset/cov.csv")
    eda_report(df)


if __name__ == "__main__":
    main()
