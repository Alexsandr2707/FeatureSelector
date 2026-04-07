import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr
# from tqdm.notebook import tqdm


def tqdm(x): return x


class CorrelationLag(TransformerMixin):
    def __init__(self, maxlag=24, minlag=0, blur=False, dropna=False,
                 corr="spearman", pvalue_output=False):
        self.maxlag = maxlag
        self.minlag = minlag
        self.blur = blur
        self.corr = corr
        self.dropna = dropna
        self.blur = blur
        self.pvalue_output = pvalue_output

    def fit(self, X, y):
        bonus = "'s p-value" if self.pvalue_output else ""
        print(f"Computing best lag using {self.corr} correlation{bonus}...")
        method = self.corr
        if self.pvalue_output:
            methodr = eval(method + "r")
            def method(x, y): return 1 - methodr(x, y).pvalue
        # assert len(y) == y.size(), "y must be 1-dimensional!"
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X)
        if type(y) != pd.DataFrame:
            y = pd.DataFrame(y)
            # y.columns = ["target"]
        self.feature_names_in_ = list(X.columns)
        self.lags = np.zeros((self.maxlag-self.minlag, X.shape[1]))
        for i in tqdm(range(self.minlag, self.maxlag)):
            d = X.drop(y.columns, axis=1).shift(i).join(y)
            corr = d.dropna().corr(method=method)
            self.lags[i-self.minlag] = corr.loc[self.feature_names_in_,
                                                y.columns[0]].values
        self.lags = pd.DataFrame(columns=self.feature_names_in_, data=self.lags,
                                 index=range(self.minlag, self.maxlag))
        if self.blur:
            if type(self.blur) == bool:
                blur = int(self.maxlag/3)
            else:
                blur = self.blur
            if blur > 2:
                self.lags = self.lags.apply(savgol_filter,
                                            window_length=blur, polyorder=2)
        return self

    def transform(self, X, y=None):
        result = {}
        lags = self.lags.idxmax(axis=0)
        for col in X.columns:
            result[col] = X[col].shift(int(lags[col]))
        result = pd.DataFrame(result)
        if self.dropna:
            result = result.dropna()
        if y is None:
            return result
        else:
            index = result.index.intersection(y.index)
            return result.loc[index], y.loc[index]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
