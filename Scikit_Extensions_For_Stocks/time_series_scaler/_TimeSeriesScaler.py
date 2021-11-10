import numpy
from sklearn.base import BaseEstimator,TransformerMixin
from copy import deepcopy

class LinearAutoRegressiveScaler(TransformerMixin,BaseEstimator):
    """
    Transformers Time Series/Linear Data by calculating AR(1) and dividing by max delta.
    """
    def __init__(self):
        self.InitialValue = None
        self.MaxDiff = None
    def __reset_state(self):
        self.InitialValue = None
        self.MaxDiff = None
    def __isInitialized(self):
        if(self.InitialValue is None or self.MaxDiff is None):
            return False
        else:
            return True
    def __partial_fit(self,X,y=None,n=1,prepend=True,forceReshape=False,**fit_params):
        """Fit the Linear Time Series Transformer on X. Expects a 1D numerical Array. 
        y will be ignored.
        Args:
            X {1D nd.array like} of shape (n,): Data for transformer to fit
            y Will be ignored.
            n (int, optional): NOT SUPPORTED . nth difference. Defaults to 1.
            prepend (bool, optional): Prepend a 0 to data. Defaults to False.
            forceReshape (bool, optional): Tries to force the shape of the array to 1D. Defaults to False.
        """
        assert n==1, "Not Supported"
        X = numpy.array(deepcopy(X)) #Paranoia
        if(forceReshape==True):
            X = numpy.ravel(X) #OMG ravel has flattened them. Will someone stop the match already?!
        assert X.ndim == 1, "Array should be 1D"
        if(self.__isInitialized()==False):
            self.InitialValue = deepcopy(X[0])
        DiffArray = numpy.diff(X, int(n))
        if(prepend==True):
            DiffArray = numpy.insert(DiffArray,[0],0,axis=None)
        if(self.__isInitialized()==False):
            self.MaxDiff = numpy.max(numpy.abs(DiffArray))
        DiffArray = DiffArray/self.MaxDiff
        return DiffArray
    
    def fit(self,X,y=None, **fit_params):
        """Fit the Linear Time Series Transformer on X. Expects a 1D numerical Array. 
        y will be ignored.
        Args:
            X {1D nd.array like} of shape (n,): Data for transformer to fit
            y Will be ignored.
            n (int, optional): NOT SUPPORTED . nth difference. Defaults to 1.
            prepend (bool, optional): Prepend a 0 to data. Defaults to False.
            forceReshape (bool, optional): Tries to force the shape of the array to 1D. Defaults to False.
        """
        self.__reset_state()
        self.__partial_fit(X, y, **fit_params)
        pass
    def transform(self,X,prepend = True,forceReshape=False):
        """
        Transform 1D array X.

        Args:
            X {1D nd.array like} of shape (n,): Data for transformer to transform
            forceReshape (bool, optional): [description]. Defaults to False.
        """
        return self.__partial_fit(X,y=None,n=1,prepend=prepend, forceReshape=forceReshape)
    def fit_transform(self, X, y=None,prepend = True,forceReshape=False, **fit_params):
        if y is None:
            self.fit(X, **fit_params)
            return self.transform(X,prepend=prepend,forceReshape=forceReshape)
        else:
            self.fit(X, y, **fit_params)
            return self.transform(X,prepend=prepend,forceReshape=forceReshape)
    
    def inverse_transform(self,X,prepend=False):
        arr = deepcopy(X)
        arr = arr*self.MaxDiff
        if(prepend==False):
            arr =  numpy.insert(arr,[0],0,axis=None)
        else:
            arr[0] = self.InitialValue
        arr = numpy.cumsum(arr)
        return arr

class Linear2DAutoRegressiveScaler():
    def __init__(self):
        self.Transformers = []
    def fit(self, X, y=None,axis=-1,**fit_params):
        """Fit the Linear Auto Regressive Transformer on X. Expects a 1D numerical Array. 
        y will be ignored.
        Args:
            X {1D nd.array like} of shape (n,): Data for transformer to fit
            y Will be ignored.
            n (int, optional): NOT SUPPORTED . nth difference. Defaults to 1.
            prepend (bool, optional): Prepend a 0 to data. Defaults to False.
            forceReshape (bool, optional): Tries to force the shape of the array to 1D. Defaults to False.
        """
        X = numpy.array(deepcopy(X))
        for val in numpy.rollaxis(X,axis):
            LAR = LinearAutoRegressiveScaler()
            LAR.fit(val)
            self.Transformers.append(LAR)
    def transform(self,X,y=None,axis=-1,prepend = True,forceReshape=False):
        """[summary]

        Args:
            X ([type]): [description]
            y ([type], optional): [description]. Defaults to None.
        """
        X = numpy.array(deepcopy(X))
        Res = []
        assert len(self.Transformers) == X.shape[axis]
        n = X.shape[axis]
        for ind,val in zip(range(0,n),numpy.rollaxis(X,axis)):
            Res.append(self.Transformers[ind].transform(val,prepend=prepend,forceReshape=forceReshape))
        return numpy.array(Res)
    def fit_transform(self, X, y=None,axis=-1,prepend = True,forceReshape=False, **fit_params):
        X = numpy.array(deepcopy(X))
        n = X.shape[axis]
        Res = []
        for val in numpy.rollaxis(X,axis):
            LAR = LinearAutoRegressiveScaler()
            Res.append(LAR.fit_transform(val,y,prepend=prepend,forceReshape=forceReshape,**fit_params))
            self.Transformers.append(LAR)
        return numpy.array(Res)
