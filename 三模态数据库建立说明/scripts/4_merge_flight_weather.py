"""
merge_flight_weather.py — 用字符串 composite key 做 merge，彻底避免 dtype 冲突
"""

import os, glob, pandas as pd, numpy as np
from tqdm import tqdm

HERE      = os.path.dirname(os.path.abspath(__file__))
BTS_DIR   = os.path.join(HERE, "Aeolus_V2", "raw", "bts")
WX_DIR    = os.path.join(HERE, "Aeolus_V2", "raw", "weather")
AP_CSV    = os.path.join(HERE, "Aeolus_V2", "raw", "airports.csv")
OUT_DIR   = os.path.join(HERE, "flight_with_weather")


def load_weather():
    """加载天气，构建 composite key 用于 merge."""
    cache_files = glob.glob(os.path.join(WX_DIR, "wx_*.parquet"))
    chunks = []
    for f in tqdm(cache_files, desc="Weather"):
        try:
            wx = pd.read_parquet(f)
            if "time" in wx.columns:
                wx["time"] = pd.to_datetime(wx["time"])
            else:
                wx = wx.reset_index()
                wx["time"] = pd.to_datetime(wx["time"])
            wx["_key"] = (wx["iata_code"] + "_" +
                          wx["time"].dt.strftime("%Y-%m-%d") + "_" +
                          wx["time"].dt.hour.astype(str))
            wx = wx.rename(columns={"temp": "TEMP", "prcp": "PRCP", "wspd": "WSPD"})
            chunks.append(wx[["_key", "TEMP", "PRCP", "WSPD"]])
        except:
            continue
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def load_coords():
    ap = pd.read_csv(AP_CSV, low_memory=False)
    ap = ap.dropna(subset=["iata_code", "latitude_deg", "longitude_deg"])
    ap = ap[ap["iata_code"].str.len() == 3].drop_duplicates("iata_code")
    return ap.set_index("iata_code")[["latitude_deg", "longitude_deg"]].to_dict("index")


def process_year(year, weather, coord_map):
    files = sorted(glob.glob(os.path.join(BTS_DIR, str(year), "*", "*.csv")))
    if not files:
        return

    print(f"\nLoading {len(files)} files for {year}...")
    chunks = [pd.read_csv(f, low_memory=False) for f in tqdm(files, desc=f"Read {year}")]
    df = pd.concat(chunks, ignore_index=True)
    print(f"  {len(df):,} rows")

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df = df.dropna(subset=["FL_DATE"])

    # Build composite keys for origin and dest weather lookup
    df["_o_key"] = (df["ORIGIN"].fillna("") + "_" +
                    df["FL_DATE"].dt.strftime("%Y-%m-%d") + "_" +
                    df["CRS_DEP_TIME"].fillna(0).astype(int).astype(str).str.zfill(4).str[:2])
    df["_d_key"] = (df["DEST"].fillna("") + "_" +
                    df["FL_DATE"].dt.strftime("%Y-%m-%d") + "_" +
                    df["CRS_ARR_TIME"].fillna(0).astype(int).astype(str).str.zfill(4).str[:2])

    # Set index for weather for fast lookup
    wx_o = weather.set_index("_key").rename(columns={"TEMP": "O_TEMP", "PRCP": "O_PRCP", "WSPD": "O_WSPD"})
    wx_d = weather.set_index("_key").rename(columns={"TEMP": "D_TEMP", "PRCP": "D_PRCP", "WSPD": "D_WSPD"})

    print("  Merging origin weather...")
    for col in ["O_TEMP", "O_PRCP", "O_WSPD"]:
        df[col] = df["_o_key"].map(wx_o[col]).fillna(0.0)

    print("  Merging destination weather...")
    for col in ["D_TEMP", "D_PRCP", "D_WSPD"]:
        df[col] = df["_d_key"].map(wx_d[col]).fillna(0.0)

    # Airport coordinates
    print("  Adding airport coordinates...")
    for prefix, col in [("O_", "ORIGIN"), ("D_", "DEST")]:
        lats, lons = [], []
        for v in df[col].values:
            info = coord_map.get(v, {})
            lats.append(info.get("latitude_deg", 0.0))
            lons.append(info.get("longitude_deg", 0.0))
        df[f"{prefix}LATITUDE"] = lats
        df[f"{prefix}LONGITUDE"] = lons

    # Write per day
    col_order = [
        "FL_DATE", "OP_CARRIER", "OP_CARRIER_FL_NUM", "TAIL_NUM",
        "ORIGIN", "DEST",
        "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY", "CRS_ARR_TIME", "ARR_TIME", "ARR_DELAY",
        "TAXI_OUT", "TAXI_IN", "CANCELLED", "DIVERTED",
        "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME",
        "FLIGHTS", "DISTANCE",
        "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
        "O_TEMP", "O_PRCP", "O_WSPD", "D_TEMP", "D_PRCP", "D_WSPD",
        "O_LATITUDE", "O_LONGITUDE", "D_LATITUDE", "D_LONGITUDE",
    ]
    keep = [c for c in col_order if c in df.columns]
    print(f"  Writing {len(df['_o_key'].unique())} daily files...")
    for _, grp in tqdm(df.groupby(df["FL_DATE"].dt.date), desc=f"Write {year}"):
        dt = grp["FL_DATE"].iloc[0]
        y, m, d = str(dt.year), f"{dt.month:02d}", f"{dt.day:02d}"
        out_path = os.path.join(OUT_DIR, y, m, f"flight_with_weather_{y}_{m}_{d}.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        grp[keep].to_csv(out_path, index=False)


def main():
    import shutil
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading weather data...")
    weather = load_weather()
    print(f"  {len(weather):,} hourly rows")

    print("Loading airport coordinates...")
    coord_map = load_coords()
    print(f"  {len(coord_map)} airports")

    for year in range(2016, 2026):
        process_year(year, weather, coord_map)

    total = len(glob.glob(os.path.join(OUT_DIR, "**", "*.csv"), recursive=True))
    print(f"\n[DONE] {total} files")


if __name__ == "__main__":
    main()
