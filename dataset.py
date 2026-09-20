from collections import namedtuple

import numpy as np
import pandas as pd

import config

Split = namedtuple("Split", "values index train_end val_end test_end")


def load_matrix(mmap=True):
    if not config.MATRIX.exists():
        raise SystemExit("matrix not found, run: python preprocess.py")
    return np.load(config.MATRIX, mmap_mode="r" if mmap else None)


def load_index():
    return pd.DatetimeIndex(pd.read_parquet(config.INDEX)["timestamp"])


def series(square):
    values = np.asarray(load_matrix()[square - 1], dtype=np.float64)
    return pd.Series(values, index=load_index(), name=f"square {square}")


def top_squares(n=3):
    totals = np.asarray(load_matrix(mmap=False)).sum(axis=1)
    return (np.argsort(totals)[-n:][::-1] + 1).tolist()


def make_split(square):
    s = series(square)
    at = lambda ts: s.index.get_indexer([pd.Timestamp(ts, tz=config.TIMEZONE)])[0]
    return Split(s.to_numpy(), s.index, at(config.VAL_START),
                 at(config.TEST_START), at(config.TEST_END) + 1)


def fit_scaler(train, use_log):
    x = np.log1p(train) if use_log else train
    return use_log, float(x.mean()), float(x.std()) or 1.0


def scale(x, scaler):
    use_log, mean, std = scaler
    x = np.log1p(x) if use_log else x
    return (x - mean) / std


def unscale(z, scaler):
    use_log, mean, std = scaler
    x = z * std + mean
    return np.clip(np.expm1(x) if use_log else x, 0.0, None)


def windows(values, start, stop, seq_len=config.SEQ_LEN):
    view = np.lib.stride_tricks.sliding_window_view(values, seq_len)
    x = view[start - seq_len:start - seq_len + (stop - start)]
    return np.ascontiguousarray(x), np.ascontiguousarray(values[start:stop])
