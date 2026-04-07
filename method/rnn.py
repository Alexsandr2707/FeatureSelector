import torch.nn as nn
import torch.optim as optim
from torch import Tensor, norm

from .model import BaseModel, GRU_state
from .lag import CorrelationLag
from .evaluate import Evaluate


class RNN(BaseModel, Evaluate):
    def __init__(
        self,
        features_in,
        features_out=1,
        lag=24,
        gru=(8, 1),
        decay=0.01,
        l2=0.5,
        lr=1e-3,
        use_scheduler=True,
        min_lr=1e-4,
    ):
        Evaluate.__init__(self)
        BaseModel.__init__(self)
        self.lag = lag
        self.l2 = l2
        self.use_scheduler = use_scheduler
        self.min_lr = min_lr
        try:
            units, num_layers = gru
        except:
            units, num_layers = gru, 1

        self.feed = nn.Sequential(
            GRU_state(
                features_in,
                units,
                num_layers=num_layers,
                batch_first=True,
                last_state=True,
            ),
            # nn.BatchNorm1d(units),
            nn.Dropout(0.25),
            # nn.LeakyReLU(),
            nn.Linear(units, features_out),
            # nn.Linear(units, 8),
            # nn.Linear(8, features_out),
        )
        self.loss = nn.L1Loss()
        # self.loss = nn.MSELoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=decay)

    def penalty(self):
        for W in self.feed[0].parameters():
            break
        w = W * (1 - self.spearman)
        l2 = self.l2 * norm(w.view(-1), 2)
        return l2

    def forward(self, x):
        return self.feed(x)

    def evaluate(self, X_train, y_train, *args, device="cpu", **kwargs):
        corr = CorrelationLag(maxlag=self.lag, blur=False, dropna=True, corr="spearman")
        corr.fit(X_train[:, -1], y_train[:, -1])
        self.spearman = Tensor(corr.lags.abs().max().values).to(device)
        nn.Module.to(self, device)
        return Evaluate.evaluate(
            self, X_train, y_train, *args, lag=self.lag, device=device, **kwargs
        )

    def fit(
        self, train, valid, epochs=1, verbose=False, early_stopping_rounds=300, **kwargs
    ):
        if self.use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=self.min_lr
            )

        return BaseModel.fit(
            self,
            train=train,
            valid=valid,
            epochs=epochs,
            penalty_func=self.penalty,
            clear=None,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs
        )
