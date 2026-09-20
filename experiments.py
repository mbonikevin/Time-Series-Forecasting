import json
import platform
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import dataset
import metrics
from models import MODELS, SeasonalNaive

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})


def build(name, params):
    params = dict(params)
    use_log = params.pop("use_log", True)
    if name == "seasonal_arimax":
        params["order"] = tuple(params["order"])
    return MODELS[name](**params), use_log


def plot_forecast(index, actual, prediction, name, square):
    fig, ax = plt.subplots(figsize=(9, 2.8), constrained_layout=True)
    ax.plot(index, actual, lw=0.9, color="0.15", label="actual")
    ax.plot(index, prediction, lw=0.9, color="C3", alpha=0.85, label="predicted")
    ax.set(ylabel="internet activity",
           title=f"{name}, square {square}, 16-22 december 2013")
    ax.legend(loc="upper right", frameon=False)
    fig.autofmt_xdate()
    fig.savefig(config.FIGURES / f"fig07_{name}_square{square}.png")
    plt.close(fig)


def plot_worst_day(index, actual, predictions, square):
    frame = pd.DataFrame(predictions, index=index)
    frame["actual"] = actual
    daily = frame.groupby(frame.index.date).apply(
        lambda d: pd.Series({m: np.mean(np.abs(d[m] - d["actual"]))
                             for m in predictions}), include_groups=False)
    worst = daily.min(axis=1).idxmax()

    day = frame[frame.index.date == worst]
    fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    ax.plot(day.index, day["actual"], lw=1.4, color="0.15", label="actual")
    for name in predictions:
        ax.plot(day.index, day[name], lw=1.0, alpha=0.85, label=name)
    ax.set(ylabel="internet activity", title=f"hardest day: {worst}, square {square}")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.savefig(config.FIGURES / f"fig08_worst_day_square{square}.png")
    plt.close(fig)

    daily.index.name = "date"
    daily.to_csv(config.TABLES / f"tab07_daily_mae_square{square}.csv")


def main():
    best = json.loads((config.LOGS / "best_params.json").read_text())
    squares = dataset.top_squares(3)
    print("busiest areas:", squares)

    timings, tables = [], []
    for square in squares:
        print(f"\nsquare {square}")
        split = dataset.make_split(square)
        predictions = {"seasonal naive": SeasonalNaive().predict(
            split, None, split.val_end, split.test_end)}

        for name in MODELS:
            model, use_log = build(name, best[name])
            scaler = dataset.fit_scaler(split.values[:split.train_end], use_log)
            train_seconds = model.fit(split, scaler)

            start = time.perf_counter()
            predictions[name] = model.predict(split, scaler,
                                              split.val_end, split.test_end)
            inference_seconds = time.perf_counter() - start

            steps = split.test_end - split.val_end
            timings.append({"square": square, "model": name,
                            "train_seconds": round(train_seconds, 2),
                            "inference_seconds": round(inference_seconds, 3),
                            "ms_per_prediction": round(inference_seconds / steps * 1000, 3)})
            print(f"  {name:<16} train {train_seconds:6.1f}s  "
                  f"inference {inference_seconds:.2f}s")

        index = split.index[split.val_end:split.test_end]
        actual = split.values[split.val_end:split.test_end]
        for name in MODELS:
            plot_forecast(index, actual, predictions[name], name, square)

        table = metrics.table(predictions, actual, split.values[:split.train_end])
        table.insert(0, "square", square)
        table.to_csv(config.TABLES / f"tab06_metrics_square{square}.csv", index=False)
        tables.append(table)
        print(table.to_string(index=False))

        plot_worst_day(index, actual, predictions, square)

    pd.concat(tables).to_csv(config.TABLES / "tab06_metrics_all.csv", index=False)

    timing = pd.DataFrame(timings)
    timing.to_csv(config.LOGS / "timing_per_square.csv", index=False)
    summary = timing.groupby("model")[["train_seconds", "inference_seconds",
                                       "ms_per_prediction"]].mean().round(3).reset_index()
    summary["hardware"] = f"{platform.machine()}, cpu only"
    summary.to_csv(config.TABLES / "tab08_timing.csv", index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
