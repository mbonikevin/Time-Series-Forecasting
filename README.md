# comparative analysis of sequential models for mobile network traffic forecasting

one-step-ahead forecasting of internet traffic in milan. three models from
three different families are compared across the three busiest areas of the
city for the week of 16-22 december 2013, against a seasonal naive baseline.

## data

source: barlacchi et al., *a multi-source dataset of urban life in the city of
milan and the province of trentino*, scientific data 2, 150055 (2015).
https://doi.org/10.1038/sdata.2015.55

download: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

download the milan `sms-call-internet-mi` files, all 62 daily files covering
2013-11-01 to 2014-01-01, and unzip them into `data/raw/`:

```
data/raw/sms-call-internet-mi-2013-11-01.txt
...
data/raw/sms-call-internet-mi-2014-01-01.txt
```

that is about 20 gb of tab-separated text. `data/` is gitignored. each line is:

```
square_id  timestamp_ms  country_code  sms_in  sms_out  call_in  call_out  internet
```

## setup

python 3.10 or newer, cpu only.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## running

```bash
python preprocess.py    # 20 gb of text -> one 357 mb matrix
python eda.py           # figures 1-6, tables 1-5
python tune.py          # staged hyperparameter search
python experiments.py   # final models, 9 forecast plots, metric and timing tables
```

## modules

| file | purpose |
|---|---|
| `config.py` | paths, dataset shape, split dates |
| `preprocess.py` | streams the 62 raw files into a dense float32 matrix, records the memory benchmark |
| `dataset.py` | matrix access, chronological split, scaling, sliding windows |
| `metrics.py` | mae, rmse, mape with a low-traffic floor, mase |
| `models.py` | seasonal naive, seasonal arimax, lstm, tcn |
| `eda.py` | distribution, spatial map, five series, weekday and weekend profiles, autocorrelation, stl, stationarity |
| `tune.py` | staged search, writes the experiment log |
| `experiments.py` | final training, forecasts, metric and timing tables |

## memory management

the raw download does not fit in memory, so four decisions reduce it:

1. **stream one day at a time.** each raw file covers exactly one milan day and
   no timestamp appears in two files, so the aggregation key never crosses a
   file boundary. peak memory tracks the largest single file, not the total.
2. **read three columns of eight.** `usecols=[0, 1, 7]` means the sms and voice
   columns are never built in memory.
3. **sum over `country_code`.** about 3.3 rows share each area and interval,
   one per destination country. these are disjoint slices of the same traffic,
   so summing is valid, and it is required: a forecaster needs exactly one
   value per area per interval.
4. **store a dense matrix, not a table of rows.** in long format every value
   carries its own square id and timestamp, which costs more than the value.
   in a 10000 x 8928 `float32` matrix the position is the label. the `.npy`
   file is memory-mappable, so reading one area costs about 36 kb.

measured in `results/logs/memory.csv`.

## outputs

```
results/figures/   fig01 - fig08
results/tables/    tab01 - tab08
results/logs/      memory.csv, experiments.csv, best_params.json, timing_per_square.csv
```
