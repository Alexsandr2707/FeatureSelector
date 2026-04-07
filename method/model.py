import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from datetime import datetime

# from IPython.display import clear_output

torch.manual_seed(0)
PATH = "%s.tmp"


class Repeat(nn.Module):
    def __init__(self, lag):
        super().__init__()
        self.lag = lag

    def forward(self, x):
        x = x.view(-1, 1, x.size()[-1])
        x = x.repeat(1, self.lag, 1)
        return x


class GRU_state(nn.GRU):
    def __init__(self, *args, last_state=True, residual=False, **argv):
        super().__init__(*args, **argv)
        self.last_state = last_state
        self.residual = residual

    def forward(self, x):
        y = super().forward(x)[0]
        if self.residual:
            y = y + x
        if self.last_state:
            y = y[:, -1]
        return y


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class BaseData(torch.utils.data.Dataset):
    def __init__(self, device, *args):
        self.device = device
        self.args = args
        assert len(np.unique([len(x) for x in args])) == 1

    def __len__(self):
        return len(self.args[0])

    def __getitem__(self, idx):
        return [
            torch.Tensor(x[idx].to(torch.float32)).to(self.device) for x in self.args
        ]


class BaseModel(nn.Module):
    def count(self):
        return sum(dict((p.data_ptr(), p.numel()) for p in self.parameters()).values())

    def iterate(self, data, func, **kwargs):
        store = []
        for portion in data:
            store.append(func(*portion, **kwargs))
        return store

    def batch(self, X, y, penalty_func=None):
        self.optimizer.zero_grad()
        pred = self(X)
        loss = self.loss(pred, y)
        if penalty_func is not None:
            loss = loss + penalty_func()
        loss.backward()
        item = loss.item()
        np.isnan(item) or self.optimizer.step()
        return pred.detach(), item

    def epoch(self, data, penalty_func=None):
        self.train()
        running_loss = 0
        counter = 0
        for portion in data:
            _, loss = self.batch(*portion, penalty_func=penalty_func)
            if not np.isnan(loss):
                running_loss += loss
                counter += 1
        assert counter != 0, "Model has gone NaN! - train"
        return running_loss / counter

    def valid(self, data):
        self.eval()
        running_loss = 0
        counter = 0
        for portion in data:
            X, y = portion
            loss = self.loss(self(X), y).item()
            if not np.isnan(loss):
                running_loss += loss
                counter += 1
        assert counter != 0, "Model has gone NaN! - valid"
        return running_loss / counter

    def fit(
        self,
        train,
        valid=None,
        epochs=10,
        early_stopping_rounds=3,
        restore=True,
        clear=None,
        verbose=True,
        penalty_func=None,
    ):
        best_valid = 10e10
        counter = 0
        path = PATH % self.__class__.__name__
        for epoch in range(epochs):
            if counter > early_stopping_rounds:
                break
            if verbose and (clear is not None):
                if epoch % clear == 0:
                    clear_output(wait=True)
            start = datetime.now()
            train_ = self.epoch(
                train() if callable(train) else train, penalty_func=penalty_func
            )
            time = datetime.now() - start
            verbose and print(
                f"epoch ({time}): {epoch}, train: {train_:1.4f}, lr: {self.optimizer.param_groups[0]['lr']:.3e}",
                end=", ",
            )
            if valid is None:
                verbose and print()
                continue
            valid_ = self.valid(valid() if callable(valid) else valid)
            if valid_ < best_valid:
                counter = 0
                best_valid = valid_
                torch.save(self.state_dict(), path)
            else:
                counter += 1
            if hasattr(self, "scheduler"):
                self.scheduler.step()
            verbose and print(f"valid: {valid_:1.4f}, best_valid: {best_valid:1.4f}")
        if valid and restore:
            self.load_state_dict(torch.load(path))

    def true_pred(self, X, y):
        return y.cpu().numpy(), self(X).cpu().detach().numpy()

    def predict(self, data, **kwargs):
        self.eval()
        result = self.iterate(data, self.true_pred, **kwargs)
        return np.concatenate([x[0] for x in result]), np.concatenate(
            [x[1] for x in result]
        )
