import time

import numpy as np

import config
from dataset import scale, unscale

np.random.seed(config.SEED)


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


MODELS = {"seasonal_arimax": SeasonalARIMAX}
