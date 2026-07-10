"""
build_network_mt.py — Flight_Network，按年全局编码 + 逐日输出
  对齐原版算法 + V2 改进：
  - 全局机场映射（跨天一致）
  - 全局 TAIL_NUM 映射（跨天一致）
  - 递归 DFS + visit 机制
  - ndata['TAIL_NUM_ENC']（V2 新增）
"""
import os, sys, glob, pandas as pd, numpy as np, torch
from collections import defaultdict
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "flight_with_weather")
OUT_DIR = os.path.join(HERE, "Aeolus_V2", "dataset", "Flight_Network")
MAX_WORKERS = 1

NDATA_FEAT_NAMES = [
    'CRS_DEP_TIME', 'DEP_DELAY', 'CRS_ARR_TIME', 'ARR_DELAY', 'WHEELS_OFF',
    'DEST', 'ORIGIN',
    'O_LATITUDE', 'O_LONGITUDE', 'D_LATITUDE', 'D_LONGITUDE',
    'FLIGHTS',
    'O_TEMP', 'O_PRCP', 'O_WSPD', 'D_TEMP', 'D_PRCP', 'D_WSPD',
    'MONTH', 'DAY_OF_WEEK', 'CRS_ARR_TIME_HOUR', 'CRS_DEP_TIME_HOUR',
]

FEAT_CONCAT_NAMES = [
    'O_LATITUDE', 'O_LONGITUDE', 'D_LATITUDE', 'D_LONGITUDE',
    'FLIGHTS', 'O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD',
    'DAY_OF_WEEK', 'MONTH', 'CRS_ARR_TIME_HOUR', 'CRS_DEP_TIME_HOUR',
    'ORIGIN', 'DEST',
]

# 用于构建全局编码表的列
COLUMNS_FOR_ENCODING = ['ORIGIN', 'DEST', 'TAIL_NUM', 'CRS_DEP_TIME', 'CRS_ARR_TIME',
                         'FL_DATE', 'MONTH', 'DAY_OF_WEEK', 'O_LATITUDE', 'O_LONGITUDE',
                         'D_LATITUDE', 'D_LONGITUDE', 'FLIGHTS',
                         'O_TEMP', 'O_PRCP', 'O_WSPD', 'D_TEMP', 'D_PRCP', 'D_WSPD',
                         'DEP_DELAY', 'ARR_DELAY', 'TAXI_OUT', 'CRS_ARR_TIME_HOUR', 'CRS_DEP_TIME_HOUR']


def hhmm_to_minutes(values):
    """Convert BTS HHMM integers to minutes since midnight."""
    hhmm = pd.to_numeric(values, errors='coerce').fillna(0).astype(np.int32)
    hour = (hhmm // 100).clip(0, 23)
    minute = (hhmm % 100).clip(0, 59)
    return (hour * 60 + minute).astype(np.int32)


def build_year_maps(year):
    """扫描一年数据，构建全局机场映射和 TAIL_NUM 映射"""
    year_files = sorted(glob.glob(os.path.join(SRC_DIR, str(year), '*', '*.csv')))
    all_aps = set()
    all_tails = set()

    for f in year_files:
        df = pd.read_csv(f, usecols=['ORIGIN', 'DEST', 'TAIL_NUM'], low_memory=False)

        for c in ['ORIGIN', 'DEST']:
            if c in df.columns:
                all_aps.update(df[c].dropna().unique())

        if 'TAIL_NUM' in df.columns:
            df['TAIL_NUM'] = df['TAIL_NUM'].astype(str).str.strip()
            all_tails.update(df['TAIL_NUM'][df['TAIL_NUM'] != ''].unique())

    aps = sorted(all_aps)
    ap_map = {a: i for i, a in enumerate(aps)}
    tails = sorted(all_tails)
    tail_map = {t: i+1 for i, t in enumerate(tails)}
    return ap_map, tail_map


def process_one_day(csv_path, ap_map, tail_map, overwrite=False):
    """处理单个文件，使用全局编码"""
    rel = os.path.relpath(csv_path, SRC_DIR).replace('\\', '/')
    parts = rel.split('/')
    if len(parts) != 3:
        return None
    yr, mo, fname = parts
    yymmdd = fname.replace('flight_with_weather_', '').replace('.csv', '').replace('_','')[2:8]
    out_path = os.path.join(OUT_DIR, yr, mo, f'flight_network_{yymmdd}.dgl')
    if os.path.exists(out_path) and not overwrite:
        return yymmdd

    df = pd.read_csv(csv_path, low_memory=False)
    n = len(df)
    if n < 2:
        return None

    # 保持 HHMM 整数用于兼容原始字段，同时另存真实分钟数用于建边。
    for c in ['CRS_DEP_TIME', 'CRS_ARR_TIME']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.int32)
    df['_dep_min'] = hhmm_to_minutes(df['CRS_DEP_TIME'])
    df['_arr_min'] = hhmm_to_minutes(df['CRS_ARR_TIME'])

    # 全局编码
    df['ORIGIN_INT'] = df['ORIGIN'].map(ap_map).fillna(0).astype(np.int64)
    df['DEST_INT'] = df['DEST'].map(ap_map).fillna(0).astype(np.int64)
    df['TAIL_NUM'] = df['TAIL_NUM'].astype(str).str.strip()
    df['TAIL_NUM_ENC'] = df['TAIL_NUM'].map(tail_map).fillna(0).astype(np.int16)

    # 预备特征值
    df['_dep_s'] = df['CRS_DEP_TIME'].astype(np.float32)
    df['_arr_s'] = df['CRS_ARR_TIME'].astype(np.float32)
    df['_dep_delay_s'] = df['DEP_DELAY'].fillna(0).astype(np.float32)
    df['_arr_delay_s'] = df['ARR_DELAY'].fillna(0).astype(np.float32)
    df['_wheels_off_s'] = (df['TAXI_OUT'].fillna(0) * 60).astype(np.float32)

    for c in ['CRS_ARR_TIME_HOUR', 'CRS_DEP_TIME_HOUR']:
        if c not in df.columns:
            df[c] = (df[c.replace('_HOUR', '')] // 100).astype(np.float32)

    import dgl

    # 出发索引（按 ORIGIN 机场分组）
    dep_idx = defaultdict(list)
    for ii in range(n):
        dep_idx[df.at[df.index[ii], 'ORIGIN_INT']].append(
            (ii, df.at[df.index[ii], '_dep_min']))
    dep_data = {}
    for ap, lst in dep_idx.items():
        lst.sort(key=lambda x: x[1])
        dep_data[ap] = (np.array([x[0] for x in lst], dtype=np.int64),
                        np.array([x[1] for x in lst], dtype=np.int32))

    # 迭代栈 DFS（替代递归，避免栈溢出，更快）
    g = dgl.DGLGraph()
    idx_arr = np.full(n, -1, dtype=np.int64)
    visit = np.zeros(n, dtype=bool)
    edge_src, edge_dst = [], []
    edge_interval, edge_aircraft, edge_airport = [], [], []

    # 预计算 15 分钟窗口，使用分钟数而不是 HHMM 差值。
    arr_min = df['_arr_min'].values
    window_end = arr_min + 15

    for root in range(n):
        if idx_arr[root] == -1:
            g.add_nodes(1)
            idx_arr[root] = g.num_nodes() - 1

        if visit[root]:
            continue

        # 迭代栈
        stack = [root]
        while stack:
            pos = stack.pop()
            if visit[pos]:
                continue
            visit[pos] = True

            row = df.iloc[pos]
            arr_t = row['_arr_min']
            if pd.isna(arr_t) or np.isnan(arr_t):
                continue
            arr_t = int(arr_t)
            dest_ap = int(row['DEST_INT'])
            if dest_ap not in dep_data:
                continue

            da, dt = dep_data[dest_ap]
            arr_plus_15 = int(window_end[pos])
            mask = (dt >= arr_t) & (dt <= arr_plus_15)
            for pos2 in np.where(mask)[0]:
                ci = da[pos2]
                if ci == pos:
                    continue
                if idx_arr[ci] == -1:
                    g.add_nodes(1)
                    idx_arr[ci] = g.num_nodes() - 1
                edge_src.append(idx_arr[pos])
                edge_dst.append(idx_arr[ci])
                delta = float(dt[pos2] - arr_t)
                edge_interval.append(delta)
                edge_aircraft.append(int(row['TAIL_NUM_ENC']))
                edge_airport.append(int(dest_ap))
                if not visit[ci]:
                    stack.append(ci)

    # 补齐未加入节点
    for i in range(n):
        if idx_arr[i] == -1:
            g.add_nodes(1)
            idx_arr[i] = g.num_nodes() - 1

    final_n = g.num_nodes()

    # ndata
    for f in NDATA_FEAT_NAMES:
        g.ndata[f] = torch.ones(final_n, 1, dtype=torch.float32)

    mid = {name: torch.zeros(final_n, 1, dtype=torch.float32) for name in NDATA_FEAT_NAMES}
    for pos in range(n):
        ni = idx_arr[pos]
        row = df.iloc[pos]
        mid['CRS_DEP_TIME'][ni]       = float(row['_dep_s'])
        mid['DEP_DELAY'][ni]          = float(row['_dep_delay_s'])
        mid['CRS_ARR_TIME'][ni]        = float(row['_arr_s'])
        mid['ARR_DELAY'][ni]           = float(row['_arr_delay_s'])
        mid['WHEELS_OFF'][ni]          = float(row['_wheels_off_s'])
        mid['DEST'][ni]                = float(row['DEST_INT'])
        mid['ORIGIN'][ni]              = float(row['ORIGIN_INT'])
        mid['O_LATITUDE'][ni]          = float(row.get('O_LATITUDE', 0) or 0)
        mid['O_LONGITUDE'][ni]         = float(row.get('O_LONGITUDE', 0) or 0)
        mid['D_LATITUDE'][ni]          = float(row.get('D_LATITUDE', 0) or 0)
        mid['D_LONGITUDE'][ni]         = float(row.get('D_LONGITUDE', 0) or 0)
        mid['FLIGHTS'][ni]             = float(row.get('FLIGHTS', 0) or 0)
        mid['O_TEMP'][ni]              = float(row.get('O_TEMP', 0) or 0)
        mid['O_PRCP'][ni]              = float(row.get('O_PRCP', 0) or 0)
        mid['O_WSPD'][ni]              = float(row.get('O_WSPD', 0) or 0)
        mid['D_TEMP'][ni]              = float(row.get('D_TEMP', 0) or 0)
        mid['D_PRCP'][ni]              = float(row.get('D_PRCP', 0) or 0)
        mid['D_WSPD'][ni]              = float(row.get('D_WSPD', 0) or 0)
        mid['MONTH'][ni]               = float(row.get('MONTH', 0) or 0)
        mid['DAY_OF_WEEK'][ni]         = float(row.get('DAY_OF_WEEK', 0) or 0)
        mid['CRS_ARR_TIME_HOUR'][ni]   = float(row.get('CRS_ARR_TIME_HOUR', 0) or 0)
        mid['CRS_DEP_TIME_HOUR'][ni]   = float(row.get('CRS_DEP_TIME_HOUR', 0) or 0)

    for name in NDATA_FEAT_NAMES:
        g.ndata[name] = mid[name]

    # feat
    feat_parts = [g.ndata[name] for name in FEAT_CONCAT_NAMES]
    g.ndata['feat'] = torch.cat(feat_parts, dim=1)
    g.ndata['feat'] = torch.nan_to_num(g.ndata['feat'], nan=0.0)

    # label
    g.ndata['label'] = g.ndata['DEP_DELAY']

    # 边
    if edge_src:
        g.add_edges(edge_src, edge_dst)
        g.edata['INTERVAL_TIME'] = torch.tensor(edge_interval, dtype=torch.float32).view(-1, 1)
        g.edata['AIRCRAFT_NUM']  = torch.tensor(edge_aircraft, dtype=torch.int16).view(-1, 1)
        g.edata['AIRPORT']       = torch.tensor(edge_airport, dtype=torch.int16).view(-1, 1)

    # TAIL_NUM_ENC
    tail_t = torch.zeros(final_n, 1, dtype=torch.int16)
    for pos in range(n):
        tail_t[idx_arr[pos]] = int(df.iloc[pos]['TAIL_NUM_ENC'])
    g.ndata['TAIL_NUM_ENC'] = tail_t

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dgl.save_graphs(out_path, [g])
    return yymmdd


def process_year(year, overwrite=False):
    """按年处理：建全局编码 → 逐天构建"""
    print('Building maps for %s...' % year)
    ap_map, tail_map = build_year_maps(year)
    print('  Airports: %d, Tail numbers: %d' % (len(ap_map), len(tail_map)))

    # 保存全年尾号映射表（一年一份）
    os.makedirs(OUT_DIR, exist_ok=True)
    import json
    rev_map = {str(v): k for k, v in tail_map.items()}
    with open(os.path.join(OUT_DIR, '%d_tails.json' % year), 'w') as jf:
        json.dump(rev_map, jf)

    year_files = sorted(glob.glob(os.path.join(SRC_DIR, str(year), '*', '*.csv')))
    print('  Processing %d files...' % len(year_files))

    count = 0
    for f in tqdm(year_files, desc='Network %s' % year):
        try:
            result = process_one_day(f, ap_map, tail_map, overwrite=overwrite)
            if result is not None:
                count += 1
        except Exception as e:
            print('\n  [ERROR] %s: %s' % (f, e))

    done = len(glob.glob(os.path.join(OUT_DIR, str(year), '**', '*.dgl'), recursive=True))
    print('  Done %s: %d files' % (year, done))


def main():
    overwrite = '--overwrite' in sys.argv
    year_args = [a for a in sys.argv[1:] if a != '--overwrite']
    years = [int(y) for y in year_args] if year_args else list(range(2016, 2026))
    for year in years:
        process_year(year, overwrite=overwrite)
    print('\nAll done!')


if __name__ == '__main__':
    main()
