"""
立即扫描已有 BTS CSV 找出所有机场，对比天气缓存，补下缺失的。
"""
import os, sys, glob, time, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from meteostat import Stations, Hourly

BTS_DIR = r'd:\Daisy\Aeolus_V2\raw\bts'
WX_DIR  = r'd:\Daisy\Aeolus_V2\raw\weather'
AP_CSV  = r'd:\Daisy\Aeolus_V2\raw\airports.csv'

# 1. 扫描全部已有 CSV
print("扫描已有 BTS CSV...")
csvs = sorted(glob.glob(os.path.join(BTS_DIR, "**", "*.csv"), recursive=True))
print(f"已找到 {len(csvs)} 个 CSV 文件")

all_airports = set()
for f in csvs:
    try:
        df = pd.read_csv(f, nrows=200, usecols=lambda c: c in ("ORIGIN", "DEST"))
        for col in ("ORIGIN", "DEST"):
            if col in df.columns:
                all_airports.update(df[col].dropna().unique())
    except:
        continue

print(f"共 {len(all_airports)} 个独特机场")

# 2. 对比已有天气缓存
existing = set()
for f in glob.glob(os.path.join(WX_DIR, "wx_*.parquet")):
    iata = os.path.basename(f)[3:-8]
    existing.add(iata)

missing = sorted(all_airports - existing)
print(f"已有天气: {len(existing)} | 缺失: {len(missing)}")
print(f"缺失机场: {missing[:20]}{'...' if len(missing)>20 else ''}")

if not missing:
    print("所有机场天气已齐全！")
    sys.exit(0)

# 3. 加载坐标
ap_df = pd.read_csv(AP_CSV, low_memory=False)

def get_coord(iata):
    row = ap_df[ap_df["iata_code"] == iata]
    if row.empty:
        return None
    return float(row.iloc[0]["latitude_deg"]), float(row.iloc[0]["longitude_deg"])

# 4. 多线程下载
MAX_WORKERS = 4

def fetch_one(iata):
    out = os.path.join(WX_DIR, f"wx_{iata}.parquet")
    if os.path.isfile(out):
        return f"{iata} 已有"
    coord = get_coord(iata)
    if not coord:
        return f"{iata} 无坐标"
    lat, lon = coord
    try:
        stations = Stations()
        stations = stations.nearby(lat, lon)
        station = stations.fetch(1)
        if station.empty:
            return f"{iata} 无气象站"
        sid = station.index[0]
        data = Hourly(sid, datetime(2016,1,1), datetime(2025,12,31,23,59))
        df = data.fetch()
        if df.empty:
            return f"{iata} 无数据"
        df = df[["temp", "prcp", "wspd"]].copy()
        df.columns = ["temp", "prcp", "wspd"]
        df.index.name = "time"
        df = df.reset_index()
        df["iata_code"] = iata
        df.to_parquet(out, index=False)
        return f"{iata} OK"
    except Exception as e:
        return f"{iata} ERR: {e}"

print(f"\n多线程补下载 {len(missing)} 个缺失机场...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    fut_map = {ex.submit(fetch_one, i): i for i in missing}
    for f in as_completed(fut_map):
        print(f"  {f.result()}")

# 5. 最终统计
final_existing = set()
for f in glob.glob(os.path.join(WX_DIR, "wx_*.parquet")):
    iata = os.path.basename(f)[3:-8]
    final_existing.add(iata)
print(f"\n最终: {len(final_existing)} 个机场有天气数据")
