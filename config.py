from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"

for _d in (PROC_DIR, FIGURES, TABLES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

MATRIX = PROC_DIR / "matrix.npy"
INDEX = PROC_DIR / "index.parquet"

N_SQUARES = 10_000
SLOTS_PER_DAY = 144
N_DAYS = 62
N_SLOTS = N_DAYS * SLOTS_PER_DAY
SLOT_MS = 600_000
TIMEZONE = "Europe/Rome"

SEQ_LEN = SLOTS_PER_DAY
VAL_START = "2013-12-09 00:00"
TEST_START = "2013-12-16 00:00"
TEST_END = "2013-12-22 23:50"
REFERENCE_SQUARES = [4159, 4556]
SEED = 42
