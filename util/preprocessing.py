from typing import List
import numpy as np
from numpy.typing import NDArray 
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
        full_array:NDArray = self.df[column].to_numpy()
        tokenized_text:List[str] = self.__text_clean_func__(full_array)
        unique, counts = np.unique(
                np.hstack(tokenized_text), 
                return_counts=True
                )

        # Constructing top 20 of the most common elements in the list
        top_20:List[str]= (
                pd.Series(counts, index=unique)
                .sort_values(ascending=False)
                .head(20)
                .index
                .to_list()
                )

        # Remove elements that are not top 20 list
        for row in range(len(tokenized_text)):
            tokenized_text[row] = np.unique(tokenized_text[row])
            for index in range(len(tokenized_text[row])):
                if tokenized_text[row][index] not in top_20:
                    np.delete(tokenized_text[row], index)

        # if inplace:
        #     # self.df.drop(columns=[column], inplace=True)
        #     self.df = pd.concat([self.df, token_dummies_df], axis=1)

        # return token_dummies_df

