import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

def getNWindowColNames(col,N):
    return [(col+str(i)) for i in range(0,N)]

class SlidingWindowTransformer(TransformerMixin):
    def __init__(self,NwindowSize=10):
        super().__init__()
        assert NwindowSize >= 1, "Window Size >= 1 please"
        self.NwindowSize = NwindowSize

    def _partialTransform(self,X, y=None,col_names=None,force_reshape=False,**fit_params):
        if(col_names!=None):
            assert self.NwindowSize==len(col_names), "Column Names array should be equal to size of window"
        i=0
        new_arr = []
        X = np.array(X)
        if(force_reshape==True):
            X = np.ravel(X)
        assert X.ndim==1, "Data should be one dimensional, else pass force_reshape=True"
        while(i+self.NwindowSize<=len(X)):
            new_arr.append(X[i:i+self.NwindowSize])
            i+=1
        if(col_names==None):
            return np.array(new_arr)
        else:
            DF = pd.DataFrame(np.array(new_arr))
            DF.columns = col_names
            return DF
    def fit_transform(self,X, y=None,col_names=None,col_namePrefix=None,force_reshape=False,**fit_params):
        if(col_namePrefix != None):
            col_names = getNWindowColNames(col_namePrefix,self.NwindowSize)
        return self._partialTransform(X,y,col_names,force_reshape,**fit_params)
    
    def inverse_transform(self,X,y=None,col_name=None,**params):
        X = np.array(X)
        finalArr = []
        finalArr.extend(X[0,:].tolist())
        finalArr.extend(X[1:,-1].tolist())
        if(col_name==None):
            return np.array(finalArr)
        else:
            return pd.DataFrame(data=finalArr,columns=[col_name])