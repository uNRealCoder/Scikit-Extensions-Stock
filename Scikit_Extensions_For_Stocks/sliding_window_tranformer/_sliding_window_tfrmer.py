import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.base import TransformerMixin

class SlidingWindowTransformer(TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def _partialTransform(self,X, y=None,NwindowSize=10,col_names=None ,**fit_params):
        if(col_names!=None):
            assert NwindowSize==len(col_names), "Column Names array should be equal to size of window"
        i=0
        new_arr = []
        X = np.array(X)
        while(i+NwindowSize<=len(X)):
            new_arr.append(X[i:i+NwindowSize])
            i+=1
        if(col_names==None):
            return np.array(new_arr)
        else:
            DF = pd.DataFrame(np.array(new_arr))
            DF.columns = col_names
            return DF

    def fit_transform(self,X, y=None,NwindowSize=10,col_names=None ,**fit_params):
        return self._partialTransform(X,y,NwindowSize,col_names,**fit_params)