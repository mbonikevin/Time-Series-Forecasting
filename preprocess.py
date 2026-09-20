import time

import numpy as np
import pandas as pd
import psutil

import config


def day_start_ms(date):
    return int(date.tz_localize(config.TIMEZONE).tz_convert("UTC").value // 1_000_000)


def reduce_day(path, date):
    df = pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 1, 7],
        names=["square", "ts", "internet"],
        dtype={"square": "int16", "ts": "int64", "internet": "float32"},
    )
    values = df["internet"].fillna(0.0).to_numpy(dtype=np.float32)
    row = df["square"].to_numpy().astype(np.int64) - 1
    col = (df["ts"].to_numpy() - day_start_ms(date)) // config.SLOT_MS
    flat = row * config.SLOTS_PER_DAY + col
    day = np.bincount(flat, weights=values,
                      minlength=config.N_SQUARES * config.SLOTS_PER_DAY)
    return day.reshape(config.N_SQUARES, config.SLOTS_PER_DAY).astype(np.float32)


def naive_day_mb(path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["square", "ts", "country", "sms_in", "sms_out",
                            "call_in", "call_out", "internet"])
    return df.memory_usage(deep=True).sum() / 1e6


def main():
    files = sorted(config.RAW_DIR.glob("sms-call-internet-mi-*.txt"))
    if len(files) != config.N_DAYS:
        raise SystemExit(f"expected {config.N_DAYS} raw files, found {len(files)}")

    matrix = np.zeros((config.N_SQUARES, config.N_SLOTS), dtype=np.float32)
    peak = psutil.Process().memory_info().rss / 1e6
    start = time.perf_counter()

    for i, path in enumerate(files):
        date = pd.Timestamp(path.stem.replace("sms-call-internet-mi-", ""))
        matrix[:, i * config.SLOTS_PER_DAY:(i + 1) * config.SLOTS_PER_DAY] = \
            reduce_day(path, date)
        peak = max(peak, psutil.Process().memory_info().rss / 1e6)
        print(f"[{i + 1:2d}/{len(files)}] {date.date()}  peak {peak:.0f} mb", flush=True)

    np.save(config.MATRIX, matrix)
    index = pd.date_range(start=f"{pd.Timestamp(files[0].stem[-10:]).date()} 00:00",
                          periods=config.N_SLOTS, freq="10min", tz=config.TIMEZONE)
    pd.DataFrame({"timestamp": index}).to_parquet(config.INDEX, index=False)

    naive_mb = naive_day_mb(max(files, key=lambda f: f.stat().st_size))
    rows = [
        ("raw text on disk", sum(f.stat().st_size for f in files) / 1e6),
        ("naive one day, all columns", naive_mb),
        ("naive projected, 62 days", naive_mb * config.N_DAYS),
        ("optimised peak rss", peak),
        ("optimised matrix in memory", matrix.nbytes / 1e6),
    ]
    pd.DataFrame(rows, columns=["stage", "megabytes"]).to_csv(
        config.LOGS / "memory.csv", index=False)

    for name, mb in rows:
        print(f"{name:<30} {mb:10.1f} mb")
    print(f"built in {(time.perf_counter() - start) / 60:.1f} min, "
          f"reduction {naive_mb * config.N_DAYS / peak:.0f}x")


if __name__ == "__main__":
    main()
