import pandas as pd 
import re 


class Preprocessing:
    def __init__(self, df:pd.DataFrame) -> None:
        self.df = df
        
    def fill_missing_NA(self, column:str):
        self.df[column].fillna("missing")

    
