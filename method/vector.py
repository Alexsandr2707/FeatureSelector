import pandas as pd
import numpy as np

def sliding_window(X, y, lag, dropna=1):
    join = pd.concat([X, y], axis=1).loc[X.index]
    join = join.loc[:,~join.columns.duplicated()]
    indexer = pd.Series(data=np.arange(0, len(join.columns)), index=join.columns)
    window = np.lib.stride_tricks.sliding_window_view(join, lag, axis=0).swapaxes(1, 2)
    assert dropna >= 0
    if dropna == 0:
        index = pd.Series(data=np.ones(len(join), dtype=bool), index=join.index)[lag-1:]
    elif dropna >= 1:
        index = join.notna().min(axis=1)[lag-1:]
    if dropna > 1:
        index[np.isnan(window[:, -dropna:].max(axis=(1, 2)))] = False # drop windows containing NaN
    X = window[index][..., indexer[X.columns]]
    y = window[index][..., indexer[y.columns]]
    return X, y, index[index].index