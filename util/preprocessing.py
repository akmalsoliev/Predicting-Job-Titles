from typing import List
import re 
import numpy as np 
import pandas as pd 


class Preprocessing:
    def __init__(self, df:pd.DataFrame) -> None:
        self.df = df
        
    def fill_missing_NA(self, column:str):
        self.df[column].fillna("missing")

    def __text_clean_func__(self, array_):
        array_split = []
        for val in array_:
            split_list = val.split("|")
            split_list = [text.lower().strip() for text in split_list]
            array_split.append(split_list)

        return array_split
        
    def tokenize(self, column:str, inplace:bool):
        """
        Tokenize specified column. 
        inplace param will be utilized to remove the specified column
        """
        full_array = self.df[column].to_numpy()
        tokenized_text = self.__text_clean_func__(full_array)

        # This is still broken!!!!
        final_out = []
        for col in tokenized_text:
            final_out.append(pd.get_dummies(pd.Series(col)))
        token_dummies_df = pd.concat(final_out, axis=1)

        if inplace:
            # self.df.drop(columns=[column], inplace=True)
            self.df = pd.concat([self.df, token_dummies_df], axis=1)

        return token_dummies_df
