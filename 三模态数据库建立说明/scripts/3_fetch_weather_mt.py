"""
fetch_weather_mt.py — 多线程下载 Meteostat 天气数据（与 BTS 转换并行）

流程：
  1. 扫描 raw/bts/ 下已有的 CSV 获取机场列表
  2. 从 ourairports.com 下载 airports.csv（自动 fallback）
  3. 用 4 个线程并发下载每个机场的逐小时天气数据
  4. 保存为 airport_stations.csv + airport_weather.parquet
"""

# 修复 Windows SSL 证书验证问题：使用系统证书库
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import os, sys, time, glob, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

try:
    from meteostat import Stations, Hourly
except ImportError:
    print("[ERROR] meteostat not installed. Run: pip install meteostat")
    sys.exit(1)

HERE         = os.path.dirname(os.path.abspath(__file__))
BTS_DIR      = os.path.join(HERE, "Aeolus_V2", "raw", "bts")
WEATHER_DIR  = os.path.join(HERE, "Aeolus_V2", "raw", "weather")
OUT_STATIONS = os.path.join(WEATHER_DIR, "airport_stations.csv")
OUT_WEATHER  = os.path.join(WEATHER_DIR, "airport_weather.parquet")
AIRPORTS_CSV = os.path.join(HERE, "Aeolus_V2", "raw", "airports.csv")

MAX_WORKERS  = 4   # 并发线程数
TASK_TIMEOUT = 180  # 单个机场下载超时(秒)

START = datetime(2016, 1, 1)
END   = datetime(2025, 12, 31, 23, 59)


def ensure_airports_csv():
    """确保 airports.csv 存在，否则从 ourairports 下载."""
    if os.path.isfile(AIRPORTS_CSV):
        return pd.read_csv(AIRPORTS_CSV, low_memory=False)

    print("Downloading airports.csv from OurAirports...")
    import requests
    url = "https://davidmegginson.github.io/ourairports-data/airports.csv"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    os.makedirs(os.path.dirname(AIRPORTS_CSV), exist_ok=True)
    with open(AIRPORTS_CSV, "wb") as f:
        f.write(r.content)
    print(f"Saved {AIRPORTS_CSV}")
    return pd.read_csv(AIRPORTS_CSV, low_memory=False)


def get_airport_list():
    """扫描 raw/bts 下已有的所有 CSV，获取 ORIGIN/DEST 机场代码."""
    files = sorted(glob.glob(os.path.join(BTS_DIR, "**", "*.csv"), recursive=True))
    if not files:
        print(f"[WARN] No BTS CSV files found in {BTS_DIR}")
        return set()

    airports = set()
    for f in files:  # 全部扫描
        try:
            df = pd.read_csv(f, nrows=1000, usecols=lambda c: c in ("ORIGIN", "DEST"))
            for col in ("ORIGIN", "DEST"):
                if col in df.columns:
                    airports.update(df[col].dropna().unique())
        except Exception:
            continue
    return airports


def build_coord_map(ap_df, iata_set):
    """根据 airports.csv 和 IATA 列表构建坐标映射."""
    ap = ap_df.dropna(subset=["iata_code", "latitude_deg", "longitude_deg"])
    ap = ap[ap["iata_code"].str.len() == 3]
    ap = ap[["iata_code", "latitude_deg", "longitude_deg"]].drop_duplicates("iata_code").set_index("iata_code")
    coord_map = {}
    missing = []
    for iata in iata_set:
        if iata in ap.index:
            coord_map[iata] = (float(ap.loc[iata, "latitude_deg"]), float(ap.loc[iata, "longitude_deg"]))
        else:
            missing.append(iata)
    if missing:
        print(f"  {len(missing)} airports missing coords (e.g. {missing[:10]})")
    return coord_map


def fetch_one_airport(iata, lat, lon):
    """下载一个机场的天气数据，返回 (iata, station_id, df 或 None)."""
    cache = os.path.join(WEATHER_DIR, f"wx_{iata}.parquet")
    if os.path.isfile(cache):
        wx = pd.read_parquet(cache)
        return iata, None, wx

    try:
        stations = Stations()
        stations = stations.nearby(lat, lon)
        station = stations.fetch(1)
        if station.empty:
            return iata, None, None

        station_id = station.index[0]
        data = Hourly(station_id, START, END)
        df = data.fetch()
        if df.empty:
            return iata, station_id, None

        df = df[["temp", "prcp", "wspd"]].copy()
        df.columns = ["temp", "prcp", "wspd"]
        df.index.name = "time"
        df = df.reset_index()
        df["iata_code"] = iata

        os.makedirs(WEATHER_DIR, exist_ok=True)
        df.to_parquet(cache, index=False)
        return iata, station_id, df
    except Exception as e:
        return iata, None, None


def main():
    os.makedirs(WEATHER_DIR, exist_ok=True)

    # 1. 获取机场列表
    print("Reading airport list from BTS CSV files...")
    iata_set = get_airport_list()
    print(f"Found {len(iata_set)} unique airports")

    if not iata_set:
        print("[ERROR] No airports found. Wait for BTS conversion or check paths.")
        sys.exit(1)

    # 2. 加载机场坐标
    print("Loading airport coordinates...")
    ap_df = ensure_airports_csv()
    coord_map = build_coord_map(ap_df, iata_set)
    print(f"Coordinates available for {len(coord_map)}/{len(iata_set)} airports")

    # 3. 多线程下载天气
    tasks = [(iata, lat, lon) for iata, (lat, lon) in coord_map.items()]
    print(f"Downloading weather data for {len(tasks)} airports ({MAX_WORKERS} threads)...")

    station_records = []
    all_weather = []

    pending = set()  # 跟踪未完成的机场
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut_map = {ex.submit(fetch_one_airport, i, la, lo): i for i, la, lo in tasks}
        pending = set(fut_map.values())  # 所有机场 iata
        for f in tqdm(as_completed(fut_map), total=len(fut_map), desc="Weather"):
            iata = fut_map[f]
            try:
                iata, sid, wx = f.result(timeout=TASK_TIMEOUT)
            except Exception as e:
                tqdm.write(f"  [TIMEOUT/ERR] {iata}: {type(e).__name__}")
                f.cancel()
                continue
            if iata in pending:
                pending.discard(iata)
            if sid:
                station_records.append({"iata_code": iata, "station_id": sid})
            if wx is not None and not wx.empty:
                all_weather.append(wx)
            time.sleep(0.02)  # 轻微限速避免封 IP

    if pending:
        print(f"\n[WARN] {len(pending)} airports skipped: {sorted(pending)}")

    # 4. 保存
    if station_records:
        sdf = pd.DataFrame(station_records).drop_duplicates("iata_code")
        sdf.to_csv(OUT_STATIONS, index=False)
        print(f"Stations saved: {OUT_STATIONS} ({len(sdf)} airports)")

    if all_weather:
        combined = pd.concat(all_weather, ignore_index=True)
        combined.to_parquet(OUT_WEATHER, index=False)
        print(f"Weather saved: {OUT_WEATHER} ({len(combined):,} hourly rows)")
    else:
        print("No weather data collected.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
