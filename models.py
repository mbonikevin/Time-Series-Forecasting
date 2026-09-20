import time

import numpy as np
import torch
import torch.nn as nn

import config
from dataset import scale, unscale, windows

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.set_num_threads(8)


class SeasonalNaive:
    name = "seasonal naive"

    def fit(self, split, scaler):
        return 0.0

    def predict(self, split, scaler, start, stop):
        lag = config.SLOTS_PER_DAY
        return split.values[start - lag:stop - lag]


def fourier_terms(positions, k_daily, k_weekly):
    columns = []
    for period, k in ((config.SLOTS_PER_DAY, k_daily),
                      (7 * config.SLOTS_PER_DAY, k_weekly)):
        for h in range(1, k + 1):
            angle = 2 * np.pi * h * positions / period
            columns += [np.sin(angle), np.cos(angle)]
    return np.column_stack(columns)


class SeasonalARIMAX:
    name = "seasonal arimax"

    def __init__(self, order=(3, 0, 1), k_daily=3, k_weekly=2):
        self.order = tuple(order)
        self.k_daily = k_daily
        self.k_weekly = k_weekly
        self.result = None

    def fit(self, split, scaler):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = scale(split.values[:split.train_end], scaler)
        x = fourier_terms(np.arange(split.train_end), self.k_daily, self.k_weekly)
        start = time.perf_counter()
        self.result = SARIMAX(y, exog=x, order=self.order, trend="c",
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False, maxiter=200)
        return time.perf_counter() - start

    def predict(self, split, scaler, start, stop):
        y = scale(split.values[:stop], scaler)
        x = fourier_terms(np.arange(stop), self.k_daily, self.k_weekly)
        extended = self.result.append(y[split.train_end:],
                                      exog=x[split.train_end:], refit=False)
        return unscale(np.asarray(extended.fittedvalues[-(stop - start):]), scaler)


class NeuralForecaster:
    name = "neural"

    def __init__(self, lr=1e-3, epochs=30, batch_size=128, patience=5, **kwargs):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.params = kwargs
        self.net = None

    def build(self):
        raise NotImplementedError

    def fit(self, split, scaler):
        torch.manual_seed(config.SEED)
        z = scale(split.values, scaler)
        x_train, y_train = windows(z, config.SEQ_LEN, split.train_end)
        x_val, y_val = windows(z, split.train_end, split.val_end)

        to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(-1)
        x_train, y_train = to_tensor(x_train), to_tensor(y_train)
        x_val, y_val = to_tensor(x_val), to_tensor(y_val)

        self.net = self.build()
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.HuberLoss(delta=1.0)

        best, best_state, bad = float("inf"), None, 0
        start = time.perf_counter()
        for _ in range(self.epochs):
            self.net.train()
            order = torch.randperm(len(x_train))
            for i in range(0, len(x_train), self.batch_size):
                batch = order[i:i + self.batch_size]
                optimiser.zero_grad()
                loss_fn(self.net(x_train[batch]), y_train[batch]).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimiser.step()

            self.net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net(x_val), y_val).item()

            if val_loss < best - 1e-5:
                best, bad = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return time.perf_counter() - start

    def predict(self, split, scaler, start, stop):
        z = scale(split.values, scaler)
        x, _ = windows(z, start, stop)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        self.net.eval()
        with torch.no_grad():
            out = self.net(x).squeeze(-1).numpy()
        return unscale(out, scaler)


class LSTM(NeuralForecaster):
    name = "lstm"

    def __init__(self, hidden=32, layers=1, **kwargs):
        super().__init__(hidden=hidden, layers=layers, **kwargs)

    def build(self):
        hidden, layers = self.params["hidden"], self.params["layers"]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(1, hidden, num_layers=layers, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.head(out[:, -1, :])

        return Net()


class TCN(NeuralForecaster):
    name = "tcn"

    def __init__(self, channels=32, levels=6, kernel=3, dropout=0.1, **kwargs):
        super().__init__(channels=channels, levels=levels, kernel=kernel,
                         dropout=dropout, **kwargs)

    def build(self):
        channels = self.params["channels"]
        levels = self.params["levels"]
        kernel = self.params["kernel"]
        dropout = self.params["dropout"]

        class Block(nn.Module):
            def __init__(self, c_in, c_out, dilation):
                super().__init__()
                self.pad = (kernel - 1) * dilation
                self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
                self.relu = nn.ReLU()
                self.drop = nn.Dropout(dropout)
                self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

            def forward(self, x):
                y = nn.functional.pad(x, (self.pad, 0))
                y = self.drop(self.relu(self.conv(y)))
                return y + (x if self.skip is None else self.skip(x))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                blocks, c_in = [], 1
                for i in range(levels):
                    blocks.append(Block(c_in, channels, 2 ** i))
                    c_in = channels
                self.body = nn.Sequential(*blocks)
                self.head = nn.Linear(channels, 1)

            def forward(self, x):
                y = self.body(x.transpose(1, 2))
                return self.head(y[:, :, -1])

        return Net()


MODELS = {"seasonal_arimax": SeasonalARIMAX, "lstm": LSTM, "tcn": TCN}
