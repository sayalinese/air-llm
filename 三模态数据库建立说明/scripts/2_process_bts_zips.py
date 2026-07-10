"""
process_bts_zips.py — 将 BTS ZIP 文件按天解压保存为 CSV

输出结构：
  Aeolus_V2/raw/bts/YYYY/MM/YYYY-MM-DD.csv
"""

import os, sys, zipfile, io, pandas as pd
from tqdm import tqdm

ZIP_DIR = os.path.join(os.path.dirname(__file__), "bts_2016_2025")
RAW_DIR = os.path.join(os.path.dirname(__file__), "Aeolus_V2", "raw", "bts")

KEEP_COLS = [
    "FL_DATE", "OP_CARRIER", "OP_CARRIER_FL_NUM",
    "TAIL_NUM", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY",
    "TAXI_OUT", "TAXI_IN",
    "CRS_ARR_TIME", "ARR_TIME", "ARR_DELAY",
    "CANCELLED", "DIVERTED",
    "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "FLIGHTS", "DISTANCE", "DAY_OF_WEEK", "MONTH", "DAY_OF_MONTH",
]

BTS_COL_MAP = {
    "FlightDate": "FL_DATE",
    "IATA_CODE_Reporting_Airline": "OP_CARRIER",
    "Tail_Number": "TAIL_NUM",
    "Flight_Number_Reporting_Airline": "OP_CARRIER_FL_NUM",
    "Origin": "ORIGIN",
    "Dest": "DEST",
    "CRSDepTime": "CRS_DEP_TIME",
    "DepTime": "DEP_TIME",
    "DepDelay": "DEP_DELAY",
    "TaxiOut": "TAXI_OUT",
    "TaxiIn": "TAXI_IN",
    "CRSArrTime": "CRS_ARR_TIME",
    "ArrTime": "ARR_TIME",
    "ArrDelay": "ARR_DELAY",
    "Cancelled": "CANCELLED",
    "Diverted": "DIVERTED",
    "CRSElapsedTime": "CRS_ELAPSED_TIME",
    "ActualElapsedTime": "ACTUAL_ELAPSED_TIME",
    "AirTime": "AIR_TIME",
    "Flights": "FLIGHTS",
    "Distance": "DISTANCE",
    "DayOfWeek": "DAY_OF_WEEK",
    "Month": "MONTH",
    "DayofMonth": "DAY_OF_MONTH",
}


def normalize(df):
    rename = {bts: old for bts, old in BTS_COL_MAP.items() if bts in df.columns}
    if rename:
        df = df.rename(columns=rename)
    if "OP_UNIQUE_CARRIER" in df.columns and "OP_CARRIER" not in df.columns:
        df = df.rename(columns={"OP_UNIQUE_CARRIER": "OP_CARRIER"})
    # 首选的 delay 列
    for old, new in [("DEP_DELAY_NEW", "DEP_DELAY"), ("ARR_DELAY_NEW", "ARR_DELAY")]:
        if new not in df.columns and old in df.columns:
            df[new] = df[old]
    return df


def process_one_zip(zip_path):
    base = os.path.basename(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        csvs = [n for n in z.namelist() if n.endswith(".csv")]
        if not csvs:
            return None
        raw = z.read(csvs[0])
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", low_memory=False)
    df = normalize(df)
    keep = [c for c in KEEP_COLS if c in df.columns]
    if not keep:
        return None
    return df[keep]


def save_daily_csv(df, out_base):
    if df.empty:
        return 0, 0
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df = df.dropna(subset=["FL_DATE"]).copy()
    if df.empty:
        return 0, 0

    days = df["FL_DATE"].dt.date.unique()
    saved = 0
    total_rows = 0
    for day in sorted(days):
        day_df = df[df["FL_DATE"].dt.date == day]
        y, m, d = str(day.year), f"{day.month:02d}", f"{day.day:02d}"
        out_dir = os.path.join(out_base, y, m)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{y}-{m}-{d}.csv")
        day_df.to_csv(out_file, index=False)
        saved += 1
        total_rows += len(day_df)
    return saved, total_rows


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    zips = sorted([f for f in os.listdir(ZIP_DIR)
                   if f.endswith(".zip") and os.path.getsize(os.path.join(ZIP_DIR, f)) > 0])
    print(f"Found {len(zips)} ZIP files")

    total_days, total_rows = 0, 0
    for zname in tqdm(zips, desc="ZIP→CSV"):
        zpath = os.path.join(ZIP_DIR, zname)
        try:
            df = process_one_zip(zpath)
        except Exception as e:
            tqdm.write(f"  [ERR] {zname}: {e}")
            continue
        if df is None:
            tqdm.write(f"  [SKIP] {zname}")
            continue
        days, rows = save_daily_csv(df, RAW_DIR)
        total_days += days
        total_rows += rows
        tqdm.write(f"  [OK] {zname}: {rows:,} rows -> {days} daily CSVs")

    print(f"\n[DONE] Total: {total_days:,} CSV files ({total_rows:,} rows)")
    print(f"Saved under {RAW_DIR}/YYYY/MM/")


if __name__ == "__main__":
    main()
