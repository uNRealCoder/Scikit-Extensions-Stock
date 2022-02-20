from sklearn.base import BaseEstimator,ClassifierMixin,RegressorMixin
from scikeras.wrappers import KerasClassifier, KerasRegressor

def KerasClassifierWrapper(tfModelFn,**modelParams):
    return KerasClassifier(tfModelFn,**modelParams)

def KerasClassifierRegressor(tfModelFn,**modelParams):
    return KerasRegressor(tfModelFn,**modelParams)