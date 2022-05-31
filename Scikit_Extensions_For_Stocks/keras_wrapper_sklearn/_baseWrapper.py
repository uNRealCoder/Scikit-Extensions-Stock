from sklearn.base import BaseEstimator,ClassifierMixin,RegressorMixin
try:
    from scikeras import KerasRegressor, KerasClassifier
except:
    from tensorflow.keras.wrappers.scikit_learn  import KerasRegressor, KerasClassifier

def KerasClassifierWrapper(tfModelFn,**modelParams):
    return KerasClassifier(tfModelFn,**modelParams)

def KerasClassifierRegressor(tfModelFn,**modelParams):
    return KerasRegressor(tfModelFn,**modelParams)