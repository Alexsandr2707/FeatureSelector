import json
import math
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

from method.metrics import metrics
from method.rnn import RNN
from method.vector import sliding_window
from method.pls import PLSTransformer

PARAMS = {
    "UZK": {
        "data": "data/uzk.csv",
        "lab": "data/uzk_lab.csv",
        "target": "UZK.Q.81AY00108.FINALPOINT",
        "freq": "1h",
        "time_intervals": ("2022", "2024"),
    }
}

DEFAULT_EXPERIMENT_PARAMS = {}


def print_del(text="", sep="="):
    print(text.center(80, sep))


def get_data(name="UZK"):
    X = pd.read_csv(PARAMS[name]["data"], index_col=0, parse_dates=True)
    Y = pd.read_csv(PARAMS[name]["lab"], index_col=0, parse_dates=True)
    y = Y[PARAMS[name]["target"]]
    y = pd.DataFrame(y)

    X = X.resample(PARAMS[name]["freq"]).first().dropna()
    y = y.resample(PARAMS[name]["freq"]).first().dropna()
    return X, y


def _interp(
    X: pd.DataFrame, y: pd.DataFrame, method="spline", interp_limit=48, freq="1h"
):
    interp_X = (
        X.asfreq(freq)
        .interpolate(method=method, order=3, limit_area="inside", limit=interp_limit)
        .dropna()
    )

    interp_y = (
        y.asfreq(freq)
        .interpolate(method=method, order=3, limit_area="inside", limit=interp_limit)
        .dropna()
    )

    return interp_X, interp_y


def _scale_train_test(
    X: pd.DataFrame, y: pd.DataFrame, split=0.6, scaler=StandardScaler
):
    result = tuple()
    stop = y.index[int(len(y) * split)]
    stop_plus_one = y.index[int(len(y) * split) + 1]

    for item in X, y:
        train = item[:stop]
        test = item[stop_plus_one:]

        if scaler:
            s = scaler().set_output(transform="pandas")
            train = s.fit_transform(train)
            test = s.transform(test)
        else:
            s = None

        result += (train, test, s)

    return result


def _generalize_indices(X, y, freq):
    X = X.asfreq(freq)
    y = y.asfreq(freq)
    index = X.index.intersection(y.index)
    X, y = X.loc[index], y.loc[index]
    return X, y


def _select_features(
    X: pd.DataFrame,
    y: pd.DataFrame,
    feature_select_method="pls",
    pls_depth=3,
):
    if feature_select_method == "pls":
        selector = PLSTransformer(depth=pls_depth, dropna=True)
        selector.fit(X, y)
        X, y = selector.transform(X, y)

    return X, y, selector


def _remove_global_outlers_iqr(
    s: pd.Series,
    rm_type="drop",
    k=1.5,  # "clip" | "drop"
):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    IQR = q3 - q1
    low = q1 - k * IQR
    high = q3 + k * IQR
    if rm_type == "drop":
        s = s.where((low <= s) & (s <= high))
    elif rm_type == "clip":
        s = s.clip(lower=low, upper=high)
    else:
        raise ValueError("Value of 'remove_type' must be in ['clip', 'drop']")
    return s


def _remove_local_outlers_iqr(
    s: pd.Series,
    window: int = 48,
    k: float = 1.5,
    rm_type="drop",  # "clip" | "drop"
):
    q1 = s.rolling(window, center=True).quantile(0.25)
    q3 = s.rolling(window, center=True).quantile(0.75)
    iqr = q3 - q1

    low = q1 - k * iqr
    high = q3 + k * iqr

    if rm_type == "drop":
        s = s.where((low <= s) & (s <= high))
    elif rm_type == "clip":
        s = s.clip(lower=low, upper=high)
    else:
        raise ValueError("Value of 'remove_type' must be in ['clip', 'drop']")

    return s


def _remove_outlers(
    X: pd.DataFrame,
    y: pd.DataFrame,
    rm_type="drop",  # "clip" | "drop"
    window=48,  # for local outlers remove
    k_iqr=1.5,
    X_local=True,
    y_local=False,
):
    local_remove = lambda s: _remove_local_outlers_iqr(
        s,
        rm_type=rm_type,
        window=window,
        k=k_iqr,
    )
    global_remove = lambda s: _remove_global_outlers_iqr(s, rm_type=rm_type, k=k_iqr)

    remove_X_out = local_remove if X_local else global_remove
    remove_y_out = local_remove if y_local else global_remove

    for col in X.columns:
        X[col] = remove_X_out(X[col])

    y.iloc[:, 0] = remove_y_out(y.iloc[:, 0])

    return X, y


def preprocess_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    freq="1h",
    use_interp=True,
    interp_method="spline",
    interp_limit=24,
    remove_outliers_params={
        "rm_type": "drop",
        "X_local": True,
        "y_local": False,
        "window": 48,
        "k_iqr": 1.5,
    },  # or None
    drop_intervals=None,  # List of time intervals to drop
    **args,
):

    if drop_intervals is not None:
        for start, end in drop_intervals:
            start, end = pd.to_datetime(start), pd.to_datetime(end)
            X.loc[start:end] = np.nan
            y.loc[start:end] = np.nan
        X = X.dropna(how="all").asfreq(freq)
        y = y.dropna(how="all").asfreq(freq)

    if remove_outliers_params is not None:
        X, y = _remove_outlers(X, y, **remove_outliers_params)

    if use_interp:
        X, y = _interp(X, y, freq=freq, interp_limit=interp_limit, method=interp_method)

    X, y = _generalize_indices(X, y, freq=freq)

    return X, y


def plot_features(data: pd.DataFrame, names=None, index_name=None):
    names = names if names is not None else data.columns
    index = data[index_name] if index_name is not None else data.index

    # Calculate fig sizes
    ax_count = len(names)
    ncols = math.ceil(ax_count**0.5)
    nrows = int(ax_count**0.5)
    if ncols * nrows < ax_count:
        nrows += 1

    # Plot all features
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.5)
    if ax_count == 1:
        axs = np.array([axs]).reshape((-1, 1))

    for num, name in enumerate(names):
        i, j = divmod(num, ncols)
        x, y = index, data[name]
        axs[i, j].plot(x, y, color="blue", lw=0.7)
        if ax_count <= 4:
            axs[i, j].set_title(name)
            axs[i, j].grid(which="major")
            axs[i, j].tick_params("x", rotation=20)
        else:
            short_name = name if len(name) <= 10 else "..." + name[-10:]
            axs[i, j].set_title(short_name)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    for num in range(ax_count, nrows * ncols):
        i, j = divmod(num, ncols)
        axs[i, j].set_visible(False)

    plt.show()


def train_and_evaluate_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame = None,
    y_valid: pd.DataFrame = None,
    lag=24,
    gru=(8, 1),
    l2=0.0,
    decay=0.0,
    epochs=10,
    batch=8,
    lr=1e-3,
    min_lr=1e-4,
    early_stoping=10,
    **args,
):
    X_train, y_train, train_index = sliding_window(X_train, y_train, lag=lag, dropna=1)
    if X_valid is not None and y_valid is not None:
        X_valid, y_valid, valid_index = sliding_window(
            X_valid, y_valid, lag=lag, dropna=1
        )

    X_train = torch.tensor(X_train).float()
    X_valid = torch.tensor(X_valid).float()
    y_train = torch.tensor(y_train).float()
    y_valid = torch.tensor(y_valid).float()

    model = RNN(
        features_in=X_train.shape[-1],
        lag=lag,
        gru=gru,
        decay=decay,
        l2=l2,
        lr=lr,
        use_scheduler=True,
        min_lr=min_lr,
    )

    result = model.evaluate(
        X_train,
        y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        train_index=train_index,
        valid_index=valid_index,
        verbose=True,
        batch=batch,
        epochs=epochs,
        device="cpu",
        early_stopping_rounds=early_stoping,
    )

    return result


def test_preprocess_dataset(params=DEFAULT_EXPERIMENT_PARAMS):
    # Prepare data
    X, y = get_data()
    X, y = preprocess_data(X, y, **params["preprocess"])

    print_del("Dataset Prepared, Stage 1")
    print_del("Prepared Features", sep="-")
    X.info()
    print_del("Prepared Target Value", sep="-")
    y.info()

    # Scale and train-test split data
    X_train, X_valid, _, y_train, y_valid, _ = _scale_train_test(X, y)

    # Select features
    X_train, y_train, selector = _select_features(
        X_train, y_train, **params["feature_selector"]
    )
    X_valid, y_valid = selector.transform(X_valid, y_valid)

    print_del("Dataset Prepared, Stage 2")
    print_del("Selected Features", sep="-")
    print(X_train.columns)

    # Print info about Train Dataset
    X = pd.concat([X_train, X_valid])
    y = pd.concat([y_train, y_valid])

    print_del("Prepared Features", sep="-")
    X.info()
    print_del("Prepared Target Value", sep="-")
    y.info()

    plot_features(X.dropna())
    plot_features(y.dropna())


def plot_results(result):
    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=fig)
    ax2 = plt.subplot2grid((2, 2), (1, 0), fig=fig)
    ax3 = plt.subplot2grid((2, 2), (1, 1), fig=fig)

    ax1.plot(
        result["train"]["true"], label="Train True", color="blue", alpha=0.7, lw=0.7
    )
    ax1.plot(
        result["train"]["pred"], label="Train Pred", color="orange", alpha=0.7, lw=0.7
    )
    ax1.plot(
        result["valid"]["true"], label="Valid True", color="green", alpha=0.7, lw=0.7
    )
    ax1.plot(
        result["valid"]["pred"], label="Valid Pred", color="red", alpha=0.7, lw=0.7
    )

    ax1.legend()
    ax1.grid(True)
    ax1.set_title("RNN Predictions vs True Values")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Target Value")

    train_residuals = result["train"]["true"] - result["train"]["pred"]
    valid_residuals = result["valid"]["true"] - result["valid"]["pred"]

    ax2.scatter(result["valid"]["true"], valid_residuals, alpha=0.5)
    ax2.axhline(0)  # zero line
    ax2.set_xlabel("Predicted values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs Predictions")

    stats.probplot(valid_residuals, dist="norm", plot=ax3)
    ax3.set_title("QQ-Plot of Residuals")

    plt.tight_layout()
    plt.show()


def run_experiment(params=DEFAULT_EXPERIMENT_PARAMS):
    # Prepare data
    X, y = get_data()
    X, y = preprocess_data(X, y, **params["preprocess"])

    print_del("Dataset Prepared, Stage 1")
    print_del("Prepared Features", sep="-")
    X.info()
    print_del("Prepared Target Value", sep="-")
    y.info()

    # Scale and train-test split data
    X_train, X_valid, _, y_train, y_valid, _ = _scale_train_test(X, y)

    # Select features
    X_train, y_train, selector = _select_features(
        X_train, y_train, **params["feature_selector"]
    )
    X_valid, y_valid = selector.transform(X_valid, y_valid)

    print_del("Dataset Prepared, Stage 2")
    print_del("Selected Features", sep="-")
    print(X_train.columns)

    # Print info about Train Dataset
    X = pd.concat([X_train, X_valid])
    y = pd.concat([y_train, y_valid])

    print_del("Prepared Features", sep="-")
    X.info()
    print_del("Prepared Target Value", sep="-")
    y.info()

    result = train_and_evaluate_model(
        X_train, y_train, X_valid=X_valid, y_valid=y_valid, **params["trainer"]
    )

    print("\nMetrics for Train:")
    print(metrics(**result["train"], cone=0))
    print("\nMetrics for Valid:")
    print(metrics(**result["valid"], cone=0))

    plot_results(result)


if __name__ == "__main__":

    params_files = ["configs/data_raw_params.json", "configs/data_interp_params.json"]
    params_file = "configs/data_interp_params.json"

    with open(params_file) as f:
        params = json.load(f)

    print_del("Used Params")
    print(params)
    run_experiment(params)
    # test_preprocess_dataset(params)

    pass
