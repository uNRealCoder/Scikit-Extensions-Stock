import numpy as np
import pandas
from sklearn.base import TransformerMixin, BaseEstimator
from numpy import fft
class FFT_1DTransformer(TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
    def fit_transform(self, X, y=None, Npoint=None,force_reshape=False,Real_Values=True ,**fit_params):
        X = np.array(X)
        if(force_reshape==True):
            X = np.ravel(X)
        assert X.ndim==1, "Data should be one dimensional, else pass force_reshape=True"
        if(Real_Values):
            return fft.rfft(X,Npoint,axis=-1)
        else:
            return fft.fft(X,Npoint,axis=-1)
    def inverse_transform():
        pass

class IFFT_1DTransformer(TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
    def fit_transform(self, X, y=None, Npoint=None,force_reshape=False,Real_Values=True ,**fit_params):
        X = np.array(X)
        if(force_reshape==True):
            X = np.ravel(X)
        assert X.ndim==1, "Data should be one dimensional, else pass force_reshape=True"
        if(Real_Values):
            return fft.irfft(X,Npoint,axis=-1)
        else:
            return fft.irfft(X,Npoint,axis=-1)