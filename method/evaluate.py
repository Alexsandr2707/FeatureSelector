import torch
import numpy as np
from pandas import Series, DataFrame

from .model import BaseData


class Evaluate:
    def __init__(self):
        self.EVAL_BATCH_SIZE = 1024

    def fit(self, train, valid, lr, batch_size=8, epochs=1000):
        raise NotImplementedError()

    def data_loader(
        self, x, y, batch_size=8, shuffle=True, device="cpu", base_data=BaseData
    ):
        return torch.utils.data.DataLoader(
            base_data(device, x, y),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=shuffle and (len(x) % batch_size < 10),
        )

    def evaluate(
        self,
        X_train,
        y_train,
        train_index,
        X_valid=None,
        y_valid=None,
        valid_index=None,
        transform=lambda x: x,
        base_data=BaseData,
        lag=24,
        steps=-1,
        batch=8,
        epochs=1000,
        device="cpu",
        **kwargs,
    ):
        if lag is not None:
            X_train, y_train = X_train[:, :lag], y_train[:, steps]
        assert (X_valid is None) == (y_valid is None) == (valid_index is None)
        data_loader = lambda *args, **kwargs: self.data_loader(
            *args, **kwargs, device=device, base_data=base_data
        )
        train = data_loader(X_train, y_train, batch_size=batch)
        if X_valid is None:
            valid = None
        else:
            if lag is not None:
                X_valid, y_valid = X_valid[:, :lag], y_valid[:, steps]
            valid = data_loader(
                X_valid, y_valid, shuffle=False, batch_size=self.EVAL_BATCH_SIZE
            )
        self.fit(train, valid, epochs=epochs, **kwargs)
        train = data_loader(
            X_train, y_train, shuffle=False, batch_size=self.EVAL_BATCH_SIZE
        )
        result = {}
        for subset in "train", "valid":
            try:
                true, pred = self.predict(eval(subset))
                true = Series(
                    data=transform(true).flatten(), index=eval(f"{subset}_index")
                )
                pred = Series(
                    data=transform(pred).flatten(), index=eval(f"{subset}_index")
                )
                result[subset] = DataFrame({"true": true, "pred": pred})
            except Exception as e:
                result[subset] = None
                continue
        return result
