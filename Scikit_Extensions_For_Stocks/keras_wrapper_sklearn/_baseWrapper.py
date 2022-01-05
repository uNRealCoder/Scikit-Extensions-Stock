from sklearn.base import BaseEstimator,ClassifierMixin,RegressorMixin
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor

def KerasClassifierWrapper(tfModelFn,**modelParams):
    return KerasClassifier(tfModelFn,**modelParams)

def KerasClassifierRegressor(tfModelFn,**modelParams):
    return KerasRegressor(tfModelFn,**modelParams)