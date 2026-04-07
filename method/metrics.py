from scipy.stats import pearsonr, spearmanr
from sklearn import metrics as m
import pandas as pd
import numpy as np


def metrics(true, pred, cone=3.5):
    # true, pred = data.true, data.pred
    true = true.fillna(0)
    pred = pred.fillna(0)
    lab = true
    index = lab.index.intersection(pred.index)
    lab_true, lab_pred = lab.loc[index], pred.loc[index]
    mae = m.mean_absolute_error(lab_true, lab_pred)
    mse = m.mean_squared_error(lab_true, lab_pred) ** 0.5
    mape = m.mean_absolute_percentage_error(lab_true, lab_pred)
    hinge = abs(true - pred) - cone
    hinge = np.where(hinge < 0, 0, hinge).mean()
    r2 = m.r2_score(lab_true, lab_pred)
    pearson = pearsonr(lab_true.values.flatten(), lab_pred.values.flatten())
    pvalue = pearson.pvalue
    corr = pearson.statistic
    return pd.Series(
        data=[mae, mse, mape, pvalue, corr, r2, hinge],
        index=["MAE", "rMSE", "MAPE", "Pearson (p-value)", "Pearson", "R2", "Hinge"],
    )
