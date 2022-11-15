from typing import List, Union
import numpy as np
from numpy.typing import NDArray 
import pandas as pd 


class Preprocessing:
    def __init__(self, df:pd.DataFrame) -> None:
        self.df = df
        
    def fill_missing_NA(self, column:str):
        self.df[column].fillna("missing")

    def text_clean_func(self, array_:Union[NDArray, List], sep:str="|") -> List[List[str]]:
        "This method is independent, because not all columns require such action"
        array_split:List[List[str]] = []
        for val in array_:
            split_list:List[str] = [text.lower().strip() for text in val.split("|")]
            array_split.append(split_list)
        return array_split
        
    def top_x_vals(self, array_:List[List[str]], top_vals:int=20):
        """
        This function will return the processed matrix with elements
        that are in top X occurance.
        """
        unique, counts = np.unique(
                np.hstack(array_), 
                return_counts=True
                )

        # Constructing top 20 of the most common elements in the matrix
        top_x:List[str]= (
                pd.Series(counts, index=unique)
                .sort_values(ascending=False)
                .head(top_vals)
                .index
                .to_list()
                )

        # Remove elements that are not top 20 matrix
        for row in range(len(array_)):
            for index in range(len(array_[row])):
                if array_[row][index] not in top_x:
                    np.delete(array_[row], index)

        return array_

    def tokenize(self, column:str, inplace:bool):
        """
        Tokenize specified column. 
        inplace param will be utilized to remove the specified column
        """
        full_array:NDArray[np.string_] = self.df[column].to_numpy()
        
        # Notes: Need to figure an architecture here
        array_:List[str] = self.__text_clean_func__(full_array)
        # if inplace:
        #     # self.df.drop(columns=[column], inplace=True)
        #     self.df = pd.concat([self.df, token_dummies_df], axis=1)

        # return token_dummies_df

