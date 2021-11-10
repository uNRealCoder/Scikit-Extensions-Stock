import numpy as np
import pandas as pd
from copy import deepcopy

def SlidingNWindow(arr,N,col_names=None):
    if(col_names!=None):
        assert N==len(col_names), "Column Names array should be equal to size of window"
    i=0
    N=20
    new_arr = []
    while(i+N<=len(arr)):
        new_arr.append(arr[i:i+N])
        i+=1
    if(col_names==None):
        return np.array(new_arr)
    else:
        DF = pd.DataFrame(np.array(new_arr))
        DF.columns = col_names
        return DF

def SlidingNWindowWithOutput(arr,N,col_names=None):
    data = SlidingNWindow(arr,N+1,col_names[0:-1])
    if(col_names==None):
        return data[:,:-1],data[:,-1]
    else:
        return data.iloc[:,:-1],data.iloc[:,-1]
