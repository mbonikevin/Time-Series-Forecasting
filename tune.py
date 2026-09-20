import itertools
import json

import pandas as pd

import config
import dataset
from metrics import mae, rmse
from models import SeasonalNaive, SeasonalARIMAX, LSTM, TCN

LOG = config.LOGS / "experiments.csv"
BEST = config.LOGS / "best_params.json"
records = []


def run(stage, name, cls, params, split, use_log):
    scaler = dataset.fit_scaler(split.values[:split.train_end], use_log)
    model = cls(**params)
    seconds = model.fit(split, scaler)
    prediction = model.predict(split, scaler, split.train_end, split.val_end)
    actual = split.values[split.train_end:split.val_end]
    score = mae(actual, prediction)

    records.append({"stage": stage, "model": name, "use_log": use_log, **params,
                    "val_MAE": score, "val_RMSE": rmse(actual, prediction),
                    "train_seconds": round(seconds, 2)})
    print(f"{stage:<12} {name:<16} log={use_log!s:<5} {json.dumps(params)} "
          f"mae {score:8.2f}  {seconds:6.1f}s", flush=True)
    return score


def tune_neural(name, cls, capacity_grid, split):
    print(f"\ntuning {name}")
    probe = capacity_grid[len(capacity_grid) // 2]
    scores = {flag: run("1-transform", name, cls, probe, split, flag)
              for flag in (False, True)}
    use_log = min(scores, key=scores.get)

    capacity_scores = {i: run("2-capacity", name, cls, p, split, use_log)
                       for i, p in enumerate(capacity_grid)}
    capacity = capacity_grid[min(capacity_scores, key=capacity_scores.get)]

    optimiser_scores = {}
    for lr, batch in itertools.product([3e-4, 1e-3, 3e-3], [64, 256]):
        option = {"lr": lr, "batch_size": batch}
        optimiser_scores[json.dumps(option)] = run(
            "3-optimiser", name, cls, {**capacity, **option}, split, use_log)
    optimiser = json.loads(min(optimiser_scores, key=optimiser_scores.get))

    return {**capacity, **optimiser, "use_log": use_log}


def tune_arimax(split):
    print("\ntuning seasonal arimax")
    probe = {"k_daily": 5, "k_weekly": 2, "order": (2, 0, 1)}
    scores = {flag: run("1-transform", "seasonal_arimax", SeasonalARIMAX,
                        probe, split, flag) for flag in (False, True)}
    use_log = min(scores, key=scores.get)

    best, best_score = None, float("inf")
    for k_daily, order in itertools.product([3, 5, 8],
                                            [(1, 0, 0), (2, 0, 1), (3, 0, 1)]):
        params = {"k_daily": k_daily, "k_weekly": 2, "order": order}
        score = run("2-structure", "seasonal_arimax", SeasonalARIMAX,
                    params, split, use_log)
        if score < best_score:
            best, best_score = params, score
    return {**best, "use_log": use_log}


def main():
    square = dataset.top_squares(1)[0]
    print(f"tuning on square {square}")
    split = dataset.make_split(square)

    actual = split.values[split.train_end:split.val_end]
    naive = SeasonalNaive().predict(split, None, split.train_end, split.val_end)
    records.append({"stage": "baseline", "model": "seasonal_naive", "use_log": False,
                    "val_MAE": mae(actual, naive), "val_RMSE": rmse(actual, naive),
                    "train_seconds": 0.0})
    print(f"seasonal naive mae {mae(actual, naive):.2f}")

    best = {
        "lstm": tune_neural("lstm", LSTM,
                            [{"hidden": 32, "layers": 1}, {"hidden": 64, "layers": 1},
                             {"hidden": 64, "layers": 2}, {"hidden": 128, "layers": 1}],
                            split),
        "tcn": tune_neural("tcn", TCN,
                           [{"channels": 16, "levels": 4}, {"channels": 32, "levels": 4},
                            {"channels": 32, "levels": 6}, {"channels": 64, "levels": 5}],
                           split),
        "seasonal_arimax": tune_arimax(split),
    }

    pd.DataFrame(records).to_csv(LOG, index=False)
    BEST.write_text(json.dumps(best, indent=2, default=str))
    print(f"\n{len(records)} runs logged")


if __name__ == "__main__":
    main()
