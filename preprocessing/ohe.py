from typing import List, Union
import math
import numpy as np
from numpy.typing import NDArray 
import pandas as pd 


class OneHotEncode:
    def __init__(self, df:pd.DataFrame, column:str) -> None:
        self.df = df
        self.column = column
        self.col_array = self.df[self.column].to_numpy()

    def text_split(self, sep:str="|"): 
        "This method is independent, because not all columns require such action"
        array_:List[List[str|float]] = []
        for val in self.col_array:
            split_list:List[str] = [text.lower().strip() for text in val.split(sep)]
            array_.append(split_list)
        self.col_array = array_
        
    def top_x_vals(self, top_vals:int=20):
        """
        This function will return the processed matrix with elements
        that are in top X occurance.
        """
        unique, counts = np.unique(
                np.hstack(self.col_array), 
                return_counts=True
                )

        # Constructing top X of the most common elements in the matrix
        self.top_x:List[str]= (
                pd.Series(counts, index=unique)
                .sort_values(ascending=False)
                .head(top_vals)
                .index
                .to_list()
                )

        # Remove elements that are not top 20 matrix
        for row in range(len(self.col_array)):
            for index in range(len(self.col_array[row])):
                if self.col_array[row][index] not in self.top_x:
                    np.delete(self.col_array[row], index)

        return self.col_array

    def __enforce_shape__(self): 
        max_len: int = max(len(lst) for lst in self.col_array)
        for list_ in self.col_array:
            if len(list_) < max_len:
                add_amount:int = max_len - len(list_)
                list_.extend([np.nan]*add_amount)

    def encode(self, inplace:bool=True):
        self.__enforce_shape__()
        array_conv:NDArray = np.array(self.col_array)
        
        if array_conv.ndim > 1:
            # Column needs to be fixed with unique!!!
            if self.top_x:
                unique = self.top_x
            else:
                unique: NDArray[np.str_] = np.unique(np.hstack(array_conv))

            row, col = array_conv.shape
            unique_col = len(unique)
            z_array: NDArray[np.float64] = np.zeros(shape=(row, unique_col))

            for row, list_ in enumerate(array_conv):
                for col, val in enumerate(list_):
                    val = val.lower().strip()
                    if str(val) != "nan": 
                        index:NDArray[np.intc] = np.where(np.array(unique) == val)[0]
                        assert len(index) <= 1, "Should not be more than one element"
                        z_array[row][index] = 1
            ohe_df: pd.DataFrame = pd.DataFrame(z_array, columns=unique)

        elif array_conv.ndim < 1:
            ohe_df = pd.get_dummies(self.df[self.column])

        if inplace:
            self.df.drop(columns=[self.column], inplace=True)
        self.df: pd.DataFrame = pd.concat([self.df, ohe_df], axis=1)
