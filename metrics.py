import numpy as np
import pandas as pd

import config


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y, p):
    keep = y > max(np.quantile(y, 0.05), 1e-6)
    return float(np.mean(np.abs((y[keep] - p[keep]) / y[keep])) * 100)


def mase(y, p, history, season=config.SLOTS_PER_DAY):
    scale = np.mean(np.abs(history[season:] - history[:-season]))
    return mae(y, p) / float(scale)


def table(predictions, actual, history):
    rows = [{"model": name, "MAE": mae(actual, p), "RMSE": rmse(actual, p),
             "MAPE": mape(actual, p), "MASE": mase(actual, p, history)}
            for name, p in predictions.items()]
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
