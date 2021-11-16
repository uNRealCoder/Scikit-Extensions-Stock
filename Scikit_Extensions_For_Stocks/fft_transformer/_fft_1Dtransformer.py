import numpy
import pandas
from sklearn.base import TransformerMixin

class FFT_Transformer(TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, y=y, **fit_params)