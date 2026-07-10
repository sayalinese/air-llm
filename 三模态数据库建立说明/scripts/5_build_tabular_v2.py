"""
build_tabular_v2.py 
  - airport_tier (机场等级)
  - 22 维特征对齐论文（去掉 PREV_*）
  - 性能优化: usecols 懒加载 + float32 + 多年并行
"""
import os, glob, gc
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "flight_with_weather")
OUT_DIR = os.path.join(HERE, "Aeolus_V2", "dataset", "Flight_Tabular")

READ_COLS = [
    'OP_CARRIER', 'OP_CARRIER_FL_NUM', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK',
    'FL_DATE', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME',
    'ORIGIN', 'DEST', 'TAIL_NUM', 'FLIGHTS',
    'O_TEMP', 'O_PRCP', 'O_WSPD', 'D_TEMP', 'D_PRCP', 'D_WSPD',
    'O_LATITUDE', 'O_LONGITUDE', 'D_LATITUDE', 'D_LONGITUDE', 'DEP_DELAY',
]

CAT_COLS = ['OP_CARRIER', 'OP_CARRIER_FL_NUM',
            'FL_YEAR', 'FL_MONTH', 'FL_DAY', 'FL_WEEK',
            'ORIGIN_INDEX', 'DEST_INDEX']

CONT_COLS = ['CRS_DEP_TIME_MIN', 'CRS_ARR_TIME_MIN', 'CRS_ELAPSED_TIME',
             'FLIGHTS',
             'O_TEMP', 'O_PRCP', 'O_WSPD', 'D_TEMP', 'D_PRCP', 'D_WSPD',
             'O_LATITUDE', 'O_LONGITUDE', 'D_LATITUDE', 'D_LONGITUDE']

TARGET = ['DEP_DELAY']


def load_tier_map():
    ap = pd.read_csv(os.path.join(HERE, "Aeolus_V2", "raw", "airports.csv"),
                     usecols=['iata_code', 'type'])
    ap = ap.dropna(subset=['iata_code'])
    tier_map = {}
    for _, row in ap.iterrows():
        t = str(row['type'])
        if t == 'large_airport':
            tier_map[row['iata_code']] = 3
        elif t == 'medium_airport':
            tier_map[row['iata_code']] = 2
        else:
            tier_map[row['iata_code']] = 1
    return tier_map


def process_year(args):
    year, tier_map = args
    files = sorted(glob.glob(os.path.join(SRC_DIR, str(year), '*', '*.csv')))
    if not files:
        return year, 0

    df = pd.concat([pd.read_csv(f, usecols=READ_COLS, low_memory=False) for f in files],
                   ignore_index=True)
    n_raw = len(df)

    lo, hi = df['DEP_DELAY'].quantile(0.01), df['DEP_DELAY'].quantile(0.99)
    df = df[(df['DEP_DELAY'] >= lo) & (df['DEP_DELAY'] <= hi)].copy()

    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    df = df.dropna(subset=['FL_DATE'])
    df['_ymd'] = df['FL_DATE'].dt.strftime('%Y%m%d')
    df['FL_YEAR'] = df['FL_DATE'].dt.year
    df.rename(columns={'MONTH': 'FL_MONTH', 'DAY_OF_MONTH': 'FL_DAY', 'DAY_OF_WEEK': 'FL_WEEK'}, inplace=True)
    df.drop(columns=['FL_DATE'], inplace=True)

    df['_crs_dep'] = pd.to_numeric(df['CRS_DEP_TIME'], errors='coerce').fillna(0).astype(int)
    df['_crs_arr'] = pd.to_numeric(df['CRS_ARR_TIME'], errors='coerce').fillna(0).astype(int)

    df['CRS_DEP_TIME_MIN'] = (df['_crs_dep'] // 100) * 60 + (df['_crs_dep'] % 100)
    df['CRS_ARR_TIME_MIN'] = (df['_crs_arr'] // 100) * 60 + (df['_crs_arr'] % 100)
    df.drop(columns=['_crs_dep', '_crs_arr', 'CRS_DEP_TIME', 'CRS_ARR_TIME'], inplace=True)

    df['ORIGIN_INDEX'] = df['ORIGIN'].astype(str)
    df['DEST_INDEX'] = df['DEST'].astype(str)
    df.drop(columns=['ORIGIN', 'DEST'], inplace=True)

    df['OP_CARRIER'] = df['OP_CARRIER'].astype(str)
    df['OP_CARRIER_FL_NUM'] = pd.to_numeric(df['OP_CARRIER_FL_NUM'], errors='coerce').fillna(0).astype(int)

    for c in df.select_dtypes(include='float').columns:
        df[c] = df[c].astype(np.float32)
    for c in ['FL_YEAR', 'FL_MONTH', 'FL_DAY', 'FL_WEEK', 'OP_CARRIER_FL_NUM',
              'CRS_DEP_TIME_MIN', 'CRS_ARR_TIME_MIN']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.int32)

    final = TARGET + CAT_COLS + CONT_COLS
    exist = [c for c in final if c in df.columns]
    out_cols = exist + ['TAIL_NUM']

    for ymd, grp in df.groupby('_ymd'):
        yr, mo = ymd[:4], ymd[4:6]
        out_dir = os.path.join(OUT_DIR, yr, mo)
        os.makedirs(out_dir, exist_ok=True)
        grp[out_cols].to_csv(os.path.join(out_dir, 'flight_with_weather_%s.csv' % ymd[2:]), index=False)

    gc.collect()
    return year, len(df)


def main():
    print('Loading airport tiers from OurAirports...')
    tier_map = load_tier_map()
    tiers = list(tier_map.values())
    print('  Airports: %d, Tier3(Hub):%d Tier2:%d Tier1:%d' % (
        len(tier_map), tiers.count(3), tiers.count(2), tiers.count(1)))

    args = [(year, tier_map) for year in range(2016, 2026)]
    max_workers = min(os.cpu_count() or 4, 6)
    print('  Parallel workers: %d' % max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_year, a): a[0] for a in args}
        for fut in tqdm(as_completed(futs), total=len(futs), desc='Years'):
            yr, n = fut.result()
            tqdm.write('  %d: %s rows' % (yr, format(n, ',d')))

    print('All done!')


if __name__ == '__main__':
    main()
