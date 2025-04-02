import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables, reference_variable):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]

        return X


def logarithm_transformer(data):
    return np.log(data)
