import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.tsa.seasonal import STL

import config
import dataset

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})


def save(fig, name):
    fig.savefig(config.FIGURES / name)
    plt.close(fig)
    print("figure", name)


def distribution(matrix):
    totals = np.asarray(matrix).sum(axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    ax[0].hist(totals, bins=120, color="0.35")
    ax[0].set(xlabel="total internet activity", ylabel="areas", title="linear scale")
    positive = totals[totals > 0]
    ax[1].hist(positive, bins=np.logspace(np.log10(positive.min()),
                                          np.log10(positive.max()), 120), color="0.35")
    ax[1].set_xscale("log")
    ax[1].set(xlabel="total internet activity", ylabel="areas", title="log scale")
    fig.suptitle("distribution of total internet traffic across 10,000 areas")
    save(fig, "fig01_distribution.png")

    fig, ax = plt.subplots(figsize=(4.4, 3.8), constrained_layout=True)
    im = ax.imshow(np.log10(totals.reshape(100, 100) + 1), origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label="log10 total activity")
    ax.set(title="spatial layout of total traffic", xlabel="column", ylabel="row")
    save(fig, "fig02_spatial.png")

    s = pd.Series(totals)
    pd.DataFrame({
        "statistic": ["mean", "median", "std", "skewness", "min", "max",
                      "max over median", "share held by busiest 1 per cent"],
        "value": [s.mean(), s.median(), s.std(), s.skew(), s.min(), s.max(),
                  s.max() / s.median(), s.nlargest(100).sum() / s.sum()],
    }).to_csv(config.TABLES / "tab01_distribution.csv", index=False)
    return totals


def five_series(matrix, index, top3):
    squares = top3 + config.REFERENCE_SQUARES
    labels = [f"square {s} (rank {i + 1})" for i, s in enumerate(top3)]
    labels += [f"square {s} (reference)" for s in config.REFERENCE_SQUARES]
    fortnight = slice(0, 14 * config.SLOTS_PER_DAY)
    t = index[fortnight]

    fig, axes = plt.subplots(len(squares), 1, figsize=(9, 10), sharex=True,
                             constrained_layout=True)
    for ax, square, label in zip(axes, squares, labels):
        ax.plot(t, np.asarray(matrix[square - 1])[fortnight], lw=0.7, color="0.2")
        ax.set_ylabel("activity")
        ax.set_title(label, loc="left", fontsize=9)
        for day in pd.date_range(t[0].normalize(), t[-1], freq="D"):
            if day.dayofweek >= 5:
                ax.axvspan(day, day + pd.Timedelta(days=1), color="0.9", zorder=0)
    axes[-1].set_xlabel("1 - 14 november 2013, weekends shaded")
    fig.suptitle("internet traffic, first two weeks")
    save(fig, "fig03_five_series.png")


def weekday_weekend(matrix, index, squares):
    weekend = index.dayofweek >= 5
    night = (index.hour >= 2) & (index.hour < 5)
    day = (index.hour >= 10) & (index.hour < 20)
    rows = []
    for square in squares:
        v = np.asarray(matrix[square - 1], dtype=float)
        rows.append({"square": square,
                     "weekday_mean": v[~weekend].mean(),
                     "weekend_mean": v[weekend].mean(),
                     "weekday_over_weekend": v[~weekend].mean() / v[weekend].mean(),
                     "day_over_night": v[day].mean() / v[night].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES / "tab02_weekday_weekend.csv", index=False)
    print(out.to_string(index=False))


def autocorrelation(series):
    plot_lags = 3 * config.SLOTS_PER_DAY
    a = acf(series.values, nlags=7 * config.SLOTS_PER_DAY, fft=True)
    p = pacf(series.values, nlags=config.SLOTS_PER_DAY)

    fig, ax = plt.subplots(2, 1, figsize=(9, 5.4), constrained_layout=True)
    ax[0].stem(np.arange(plot_lags + 1), a[:plot_lags + 1],
               markerfmt=" ", basefmt=" ", linefmt="0.3")
    for k in (1, 2, 3):
        ax[0].axvline(k * config.SLOTS_PER_DAY, color="0.6", ls="--", lw=0.8)
    ax[0].set(title="autocorrelation", xlabel="lag (10 minute steps)")
    ax[1].stem(np.arange(len(p)), p, markerfmt=" ", basefmt=" ", linefmt="0.3")
    ax[1].set(title="partial autocorrelation", xlabel="lag (10 minute steps)")
    fig.suptitle(f"temporal dependence, {series.name}")
    save(fig, "fig04_autocorrelation.png")

    lags = [1, 6, 36, 72, 144, 288, 1008]
    names = ["10 min", "1 hour", "6 hours", "12 hours", "1 day", "2 days", "1 week"]
    pd.DataFrame({"lag": lags, "meaning": names,
                  "acf": [a[l] for l in lags]}).to_csv(
        config.TABLES / "tab03_autocorrelation.csv", index=False)


def seasonality(series):
    stl = STL(series, period=config.SLOTS_PER_DAY, robust=True).fit()
    fig, ax = plt.subplots(4, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for axis, data, name in zip(ax, [series, stl.trend, stl.seasonal, stl.resid],
                                ["observed", "trend", "daily seasonal", "remainder"]):
        axis.plot(series.index, data, lw=0.5, color="0.2")
        axis.set_ylabel(name, fontsize=8)
    fig.suptitle(f"stl decomposition, {series.name}")
    save(fig, "fig05_stl.png")

    frame = series.to_frame("activity")
    frame["minute"] = frame.index.hour * 60 + frame.index.minute
    frame["weekend"] = frame.index.dayofweek >= 5
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    for flag, label in [(False, "weekday"), (True, "weekend")]:
        grouped = frame[frame.weekend == flag].groupby("minute")["activity"]
        mean, sd = grouped.mean(), grouped.std()
        ax.plot(mean.index / 60, mean, label=label, lw=1.2)
        ax.fill_between(mean.index / 60, mean - sd, mean + sd, alpha=0.15)
    ax.set(xlabel="hour of day", ylabel="mean activity", xticks=range(0, 25, 3),
           title=f"average daily profile, {series.name}")
    ax.legend()
    save(fig, "fig06_daily_profile.png")

    rows = []
    for label, data in [("raw", series),
                        ("first difference", series.diff().dropna()),
                        ("seasonal difference", series.diff(config.SLOTS_PER_DAY).dropna())]:
        adf_stat, adf_p, *_ = adfuller(data, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(data, regression="c", nlags="auto")
        rows.append({"series": label, "adf_stat": adf_stat, "adf_p": adf_p,
                     "kpss_stat": kpss_stat, "kpss_p": kpss_p,
                     "variance": float(np.var(data))})
    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES / "tab04_stationarity.csv", index=False)
    print(out.to_string(index=False))


def main():
    matrix = dataset.load_matrix()
    index = dataset.load_index()

    totals = distribution(matrix)
    top3 = (np.argsort(totals)[-3:][::-1] + 1).tolist()
    print("busiest areas:", top3)
    pd.DataFrame({"rank": [1, 2, 3], "square": top3,
                  "total": totals[[s - 1 for s in top3]]}).to_csv(
        config.TABLES / "tab05_top_areas.csv", index=False)

    five_series(matrix, index, top3)
    weekday_weekend(matrix, index, top3 + config.REFERENCE_SQUARES)

    busiest = pd.Series(np.asarray(matrix[top3[0] - 1], dtype=float), index=index,
                        name=f"square {top3[0]}")
    autocorrelation(busiest)
    seasonality(busiest)


if __name__ == "__main__":
    main()
