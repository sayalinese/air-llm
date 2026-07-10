"""
Build Flight_Chain day-level tensors and LLM-friendly JSONL samples.

Outputs:
- Flight_Chain/{year}/{month}/flight_chain_{yymmdd}.pt
- Flight_Chain_LLM/{year}/{month}/flight_chain_llm_{yymmdd}.jsonl
- reverse mapping assets for airport / carrier / flight number / tail number
"""

import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "flight_with_weather")
CHAIN_OUT_DIR = os.path.join(HERE, "Aeolus_V2", "dataset", "Flight_Chain")
LLM_OUT_DIR = os.path.join(HERE, "Aeolus_V2", "dataset", "Flight_Chain_LLM")
MAX_CHAIN = 6
LLM_HISTORY_DEPTH = 2
MAX_WORKERS = min(os.cpu_count() or 4, 8)
DELAY_CLIP = 240.0
SLACK_CLIP = 720.0

TASK_INSTRUCTION = "根据以下航班链信息判断当前航班是否会出发延误超过15分钟。只输出：正常 或 延误。"
LABEL_TEXT = {0: "正常", 1: "延误"}

COLUMNS = [
    "FL_DATE",
    "TAIL_NUM",
    "ORIGIN",
    "DEST",
    "O_TEMP",
    "D_TEMP",
    "O_PRCP",
    "D_PRCP",
    "O_WSPD",
    "D_WSPD",
    "FLIGHTS",
    "MONTH",
    "DAY_OF_WEEK",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "O_LATITUDE",
    "O_LONGITUDE",
    "D_LATITUDE",
    "D_LONGITUDE",
    "OP_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ARR_DELAY",
    "DEP_DELAY",
]

SCAN_COLUMNS = ["TAIL_NUM", "ORIGIN", "DEST", "OP_CARRIER", "OP_CARRIER_FL_NUM"]
BASE_DENSE_VALUE_COLS = ["O_TEMP", "D_TEMP", "O_PRCP", "D_PRCP", "O_WSPD", "D_WSPD", "FLIGHTS", "CRS_ELAPSED_TIME", "O_LATITUDE", "O_LONGITUDE", "D_LATITUDE", "D_LONGITUDE"]
DERIVED_DENSE_COLS = [
    "PREV_DEP_DELAY",
    "PREV_ARR_DELAY",
    "TURNAROUND_SLACK_MIN",
    "PREV2_DEP_DELAY_MEAN",
    "PREV2_DEP_DELAY_MAX",
    "DEP_DELAY_TREND",
]
DENSE_COLS = BASE_DENSE_VALUE_COLS + DERIVED_DENSE_COLS
SPARSE_COLS = ["FL_MONTH", "FL_WEEK", "CAH", "CDH", "OI", "DI", "OC_ENC", "FN_ENC", "TE"]

DENSE_NAME_MAP_CN = {
    "O_TEMP": "出发地气温",
    "D_TEMP": "到达地气温",
    "O_PRCP": "出发地降水",
    "D_PRCP": "到达地降水",
    "O_WSPD": "出发地风速",
    "D_WSPD": "到达地风速",
    "FLIGHTS": "航班频次",
    "CRS_ELAPSED_TIME": "计划飞行时长(分钟)",
    "O_LATITUDE": "出发机场纬度",
    "O_LONGITUDE": "出发机场经度",
    "D_LATITUDE": "到达机场纬度",
    "D_LONGITUDE": "到达机场经度",
    "PREV_DEP_DELAY": "前序出发延误分钟数",
    "PREV_ARR_DELAY": "前序到达延误分钟数",
    "TURNAROUND_SLACK_MIN": "计划过站缓冲时间(分钟)",
    "PREV2_DEP_DELAY_MEAN": "前两段出发延误均值",
    "PREV2_DEP_DELAY_MAX": "前两段出发延误最大值",
    "DEP_DELAY_TREND": "前两段出发延误变化趋势",
}

SPARSE_NAME_MAP_CN = {
    "FL_MONTH": "月份编码",
    "FL_WEEK": "星期编码",
    "CAH": "计划到达小时编码",
    "CDH": "计划出发小时编码",
    "OI": "出发机场编码",
    "DI": "到达机场编码",
    "OC_ENC": "承运人编码",
    "FN_ENC": "航班号编码",
    "TE": "飞机尾号编码",
}

ENCODERS = None


def normalize_token(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_flight_number(value):
    if pd.isna(value):
        return ""
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return str(value).strip()


def hhmm_to_hour(values):
    hhmm = pd.to_numeric(values, errors="coerce").fillna(0).astype(np.int32)
    return (hhmm // 100).clip(0, 23).astype(np.int16)


def hhmm_to_minutes(values):
    hhmm = pd.to_numeric(values, errors="coerce").fillna(0).astype(np.int32)
    hour = (hhmm // 100).clip(0, 23)
    minute = (hhmm % 100).clip(0, 59)
    return (hour * 60 + minute).astype(np.int16)


def hhmm_to_text(value):
    if pd.isna(value):
        return "缺失"
    hhmm = int(float(value))
    hour = max(0, min(23, hhmm // 100))
    minute = max(0, min(59, hhmm % 100))
    return f"{hour:02d}:{minute:02d}"


def format_value(value, digits=1):
    if pd.isna(value):
        return "缺失"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.{digits}f}"
    return str(value)


def to_small_int_series(values, default=0, min_value=None):
    series = pd.to_numeric(values, errors="coerce").fillna(default)
    if min_value is not None:
        series = series.clip(lower=min_value)
    return series.astype(np.int16)


def collect_year_files(year):
    return sorted(glob.glob(os.path.join(SRC_DIR, str(year), "*", "*.csv")))


def build_year_encoders(year_files):
    carriers, flnums, airports, tails = set(), set(), set(), set()

    print("Scanning year-level vocabularies...")
    for path in tqdm(year_files, desc="  Scan", leave=False):
        df = pd.read_csv(path, usecols=SCAN_COLUMNS, low_memory=False)
        carriers.update([normalize_token(v) for v in df["OP_CARRIER"].dropna().tolist() if normalize_token(v)])
        flnums.update([normalize_flight_number(v) for v in df["OP_CARRIER_FL_NUM"].dropna().tolist() if normalize_flight_number(v)])
        airports.update([normalize_token(v) for v in df["ORIGIN"].dropna().tolist() if normalize_token(v)])
        airports.update([normalize_token(v) for v in df["DEST"].dropna().tolist() if normalize_token(v)])
        tails.update([normalize_token(v) for v in df["TAIL_NUM"].dropna().tolist() if normalize_token(v)])

    return {
        "carrier_map": {v: i for i, v in enumerate(sorted(carriers))},
        "flnum_map": {v: i for i, v in enumerate(sorted(flnums))},
        "airport_map": {v: i for i, v in enumerate(sorted(airports))},
        "tail_map": {v: i + 1 for i, v in enumerate(sorted(tails))},
    }


def save_reverse_maps(year, encoders):
    os.makedirs(CHAIN_OUT_DIR, exist_ok=True)

    airport_rev = {str(v): k for k, v in encoders["airport_map"].items()}
    carrier_rev = {str(v): k for k, v in encoders["carrier_map"].items()}
    flnum_rev = {str(v): k for k, v in encoders["flnum_map"].items()}
    tail_rev = {str(v): k for k, v in encoders["tail_map"].items()}

    with open(os.path.join(CHAIN_OUT_DIR, f"{year}_airports.json"), "w", encoding="utf-8") as f:
        json.dump(airport_rev, f, ensure_ascii=False, indent=2)
    with open(os.path.join(CHAIN_OUT_DIR, f"{year}_carriers.json"), "w", encoding="utf-8") as f:
        json.dump(carrier_rev, f, ensure_ascii=False, indent=2)
    with open(os.path.join(CHAIN_OUT_DIR, f"{year}_flight_nums.json"), "w", encoding="utf-8") as f:
        json.dump(flnum_rev, f, ensure_ascii=False, indent=2)
    with open(os.path.join(CHAIN_OUT_DIR, f"{year}_tails.json"), "w", encoding="utf-8") as f:
        json.dump(tail_rev, f, ensure_ascii=False, indent=2)


def init_worker(encoders):
    global ENCODERS
    ENCODERS = encoders


def infer_output_paths(csv_path):
    rel = os.path.relpath(csv_path, SRC_DIR).replace("\\", "/")
    year, month, filename = rel.split("/")
    day_tag = filename.replace("flight_with_weather_", "").replace(".csv", "").replace("_", "")
    yymmdd = day_tag[2:8]
    pt_dir = os.path.join(CHAIN_OUT_DIR, year, month)
    pt_path = os.path.join(pt_dir, f"flight_chain_{yymmdd}.pt")
    llm_dir = os.path.join(LLM_OUT_DIR, year, month)
    llm_path = os.path.join(llm_dir, f"flight_chain_llm_{yymmdd}.jsonl")
    return year, month, yymmdd, pt_dir, pt_path, llm_dir, llm_path


def build_derived_dense(dep_delay_seq, arr_delay_seq, dep_min_seq, arr_min_seq):
    n = len(dep_delay_seq)
    values = np.full((n, len(DERIVED_DENSE_COLS)), np.nan, dtype=np.float32)

    if n == 0:
        return values

    for i in range(1, n):
        prev_dep = float(np.clip(dep_delay_seq[i - 1], -DELAY_CLIP, DELAY_CLIP)) if not np.isnan(dep_delay_seq[i - 1]) else np.nan
        prev_arr = float(np.clip(arr_delay_seq[i - 1], -DELAY_CLIP, DELAY_CLIP)) if not np.isnan(arr_delay_seq[i - 1]) else np.nan
        slack = float(np.clip(dep_min_seq[i] - arr_min_seq[i - 1], -SLACK_CLIP, SLACK_CLIP))

        values[i, 0] = prev_dep
        values[i, 1] = prev_arr
        values[i, 2] = slack

        hist = dep_delay_seq[max(0, i - 2):i].astype(np.float32)
        hist = hist[~np.isnan(hist)]
        if hist.size > 0:
            hist = np.clip(hist, -DELAY_CLIP, DELAY_CLIP)
            values[i, 3] = float(hist.mean())
            values[i, 4] = float(hist.max())

        if i >= 2 and not np.isnan(dep_delay_seq[i - 1]) and not np.isnan(dep_delay_seq[i - 2]):
            values[i, 5] = float(np.clip(dep_delay_seq[i - 1] - dep_delay_seq[i - 2], -DELAY_CLIP, DELAY_CLIP))

    return values


def render_current_block(row):
    return "\n".join(
        [
            f"月份：{format_value(row['MONTH'])}",
            f"星期：{format_value(row['DAY_OF_WEEK'])}",
            f"计划出发时间：{hhmm_to_text(row['CRS_DEP_TIME'])}",
            f"计划到达时间：{hhmm_to_text(row['CRS_ARR_TIME'])}",
            f"出发机场：{normalize_token(row['ORIGIN']) or '缺失'}",
            f"到达机场：{normalize_token(row['DEST']) or '缺失'}",
            f"承运人：{normalize_token(row['OP_CARRIER']) or '缺失'}",
            f"航班号：{normalize_flight_number(row['OP_CARRIER_FL_NUM']) or '缺失'}",
            f"飞机尾号：{normalize_token(row['TAIL_NUM']) or '缺失'}",
            f"出发地气温：{format_value(row['O_TEMP'])}",
            f"到达地气温：{format_value(row['D_TEMP'])}",
            f"出发地降水：{format_value(row['O_PRCP'])}",
            f"到达地降水：{format_value(row['D_PRCP'])}",
            f"出发地风速：{format_value(row['O_WSPD'])}",
            f"到达地风速：{format_value(row['D_WSPD'])}",
            f"航班频次：{format_value(row['FLIGHTS'])}",
            f"计划飞行时长(分钟)：{format_value(row['CRS_ELAPSED_TIME'])}",
            f"出发机场纬度：{format_value(row['O_LATITUDE'])}",
            f"出发机场经度：{format_value(row['O_LONGITUDE'])}",
            f"到达机场纬度：{format_value(row['D_LATITUDE'])}",
            f"到达机场经度：{format_value(row['D_LONGITUDE'])}",
        ]
    )


def render_history_block(row):
    return "\n".join(
        [
            f"出发机场：{normalize_token(row['ORIGIN']) or '缺失'}",
            f"到达机场：{normalize_token(row['DEST']) or '缺失'}",
            f"承运人：{normalize_token(row['OP_CARRIER']) or '缺失'}",
            f"航班号：{normalize_flight_number(row['OP_CARRIER_FL_NUM']) or '缺失'}",
            f"飞机尾号：{normalize_token(row['TAIL_NUM']) or '缺失'}",
            f"计划出发时间：{hhmm_to_text(row['CRS_DEP_TIME'])}",
            f"计划到达时间：{hhmm_to_text(row['CRS_ARR_TIME'])}",
            f"实际出发延误分钟数：{format_value(row['DEP_DELAY'])}",
            f"实际到达延误分钟数：{format_value(row['ARR_DELAY'])}",
        ]
    )


def render_llm_text(current_row, prev_rows, derived_vals):
    sections = ["当前航班：", render_current_block(current_row)]

    for idx in range(LLM_HISTORY_DEPTH):
        if idx < len(prev_rows):
            sections.extend([f"前序航班{idx + 1}：", render_history_block(prev_rows[-(idx + 1)])])
        else:
            sections.extend([f"前序航班{idx + 1}：", "无前序航班"])

    summary = [
        f"前序出发延误分钟数：{format_value(derived_vals[0])}",
        f"前序到达延误分钟数：{format_value(derived_vals[1])}",
        f"计划过站缓冲时间(分钟)：{format_value(derived_vals[2])}",
        f"前两段出发延误均值：{format_value(derived_vals[3])}",
        f"前两段出发延误最大值：{format_value(derived_vals[4])}",
        f"前两段出发延误变化趋势：{format_value(derived_vals[5])}",
    ]
    sections.extend(["传播摘要：", "\n".join(summary)])
    return "\n".join(sections)


def process_day_file(csv_path, overwrite=False):
    year, month, yymmdd, pt_dir, pt_path, llm_dir, llm_path = infer_output_paths(csv_path)
    if os.path.exists(pt_path) and os.path.exists(llm_path) and not overwrite:
        return {"file": csv_path, "chains": None, "samples": None, "skipped": True}

    df = pd.read_csv(csv_path, usecols=COLUMNS, low_memory=False)
    if df.empty:
        return {"file": csv_path, "chains": 0, "samples": 0, "skipped": False}

    df = df.dropna(subset=["TAIL_NUM"]).copy()
    df["TAIL_NUM"] = df["TAIL_NUM"].map(normalize_token)
    df = df[df["TAIL_NUM"] != ""].copy()
    if len(df) < 2:
        return {"file": csv_path, "chains": 0, "samples": 0, "skipped": False}

    df["CDH"] = hhmm_to_hour(df["CRS_DEP_TIME"])
    df["CAH"] = hhmm_to_hour(df["CRS_ARR_TIME"])
    df["CRS_DEP_TIME_MIN"] = hhmm_to_minutes(df["CRS_DEP_TIME"])
    df["CRS_ARR_TIME_MIN"] = hhmm_to_minutes(df["CRS_ARR_TIME"])
    df["FL_MONTH"] = to_small_int_series(df["MONTH"], default=1, min_value=1) - 1
    df["FL_WEEK"] = to_small_int_series(df["DAY_OF_WEEK"], default=1, min_value=1) - 1

    for col in BASE_DENSE_VALUE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    dep_delay = pd.to_numeric(df["DEP_DELAY"], errors="coerce").astype(np.float32)
    arr_delay = pd.to_numeric(df["ARR_DELAY"], errors="coerce").astype(np.float32)
    df["DEP_DELAY_FILLED"] = dep_delay
    df["ARR_DELAY_FILLED"] = arr_delay
    df["DEP_DELAY_BIN"] = (dep_delay > 15).fillna(False).astype(np.int8)
    df["_dep_sort"] = pd.to_numeric(df["CRS_DEP_TIME"], errors="coerce").fillna(0).astype(np.int32)
    df["OP_CARRIER_NORM"] = df["OP_CARRIER"].map(normalize_token)
    df["OP_CARRIER_FL_NUM_NORM"] = df["OP_CARRIER_FL_NUM"].map(normalize_flight_number)
    df["ORIGIN_NORM"] = df["ORIGIN"].map(normalize_token)
    df["DEST_NORM"] = df["DEST"].map(normalize_token)

    carrier_map = ENCODERS["carrier_map"]
    flnum_map = ENCODERS["flnum_map"]
    airport_map = ENCODERS["airport_map"]
    tail_map = ENCODERS["tail_map"]

    df["OC_ENC"] = df["OP_CARRIER_NORM"].map(carrier_map).fillna(0).astype(np.int16)
    df["FN_ENC"] = df["OP_CARRIER_FL_NUM_NORM"].map(flnum_map).fillna(0).astype(np.int16)
    df["OI"] = df["ORIGIN_NORM"].map(airport_map).fillna(0).astype(np.int16)
    df["DI"] = df["DEST_NORM"].map(airport_map).fillna(0).astype(np.int16)
    df["TE"] = df["TAIL_NUM"].map(tail_map).fillna(0).astype(np.int16)

    df = df.sort_values(["TAIL_NUM", "_dep_sort"], kind="mergesort").reset_index(drop=True)
    group_indices = df.groupby("TAIL_NUM", sort=False).indices
    num_chains = sum(1 for idx in group_indices.values() if len(idx) >= 2)
    if num_chains == 0:
        return {"file": csv_path, "chains": 0, "samples": 0, "skipped": False}

    base_dense_day = df[BASE_DENSE_VALUE_COLS].to_numpy(dtype=np.float32, copy=False)
    sparse_day = df[SPARSE_COLS].to_numpy(dtype=np.int16, copy=False)
    labels_day = df["DEP_DELAY_BIN"].to_numpy(dtype=np.int8, copy=False)
    dep_delays_day = np.clip(df["DEP_DELAY_FILLED"].to_numpy(dtype=np.float32, copy=False), -DELAY_CLIP, DELAY_CLIP)
    arr_delays_day = np.clip(df["ARR_DELAY_FILLED"].to_numpy(dtype=np.float32, copy=False), -DELAY_CLIP, DELAY_CLIP)
    dep_minutes_day = df["CRS_DEP_TIME_MIN"].to_numpy(dtype=np.int16, copy=False)
    arr_minutes_day = df["CRS_ARR_TIME_MIN"].to_numpy(dtype=np.int16, copy=False)

    dense_out = np.full((num_chains, MAX_CHAIN, len(DENSE_COLS)), np.nan, dtype=np.float32)
    sparse_out = np.zeros((num_chains, MAX_CHAIN, len(SPARSE_COLS)), dtype=np.int16)
    labels_out = np.zeros((num_chains, MAX_CHAIN, 1), dtype=np.int8)
    delays_out = np.full((num_chains, MAX_CHAIN, 1), np.nan, dtype=np.float32)
    vlen_out = np.zeros((num_chains,), dtype=np.int64)
    llm_records = []

    chain_idx = 0
    for idx in group_indices.values():
        if len(idx) < 2:
            continue
        take = min(len(idx), MAX_CHAIN)
        pos = idx[:take]
        chain_df = df.iloc[pos].reset_index(drop=True)

        derived_vals = build_derived_dense(
            dep_delays_day[pos],
            arr_delays_day[pos],
            dep_minutes_day[pos],
            arr_minutes_day[pos],
        )
        dense_features = np.concatenate([base_dense_day[pos], derived_vals], axis=1)

        dense_out[chain_idx, :take] = dense_features
        sparse_out[chain_idx, :take] = sparse_day[pos]
        labels_out[chain_idx, :take, 0] = labels_day[pos]
        delays_out[chain_idx, :take, 0] = dep_delays_day[pos]
        vlen_out[chain_idx] = take

        for step_idx in range(take):
            current_row = chain_df.iloc[step_idx]
            prev_rows = [chain_df.iloc[j] for j in range(max(0, step_idx - LLM_HISTORY_DEPTH), step_idx)]
            llm_text = render_llm_text(current_row, prev_rows, derived_vals[step_idx])
            label = int(labels_day[pos][step_idx])
            llm_records.append(
                {
                    "sample_id": f"{yymmdd}_{chain_idx:06d}_{step_idx + 1}",
                    "date": str(current_row["FL_DATE"]),
                    "chain_position": int(step_idx + 1),
                    "chain_valid_length": int(take),
                    "instruction": TASK_INSTRUCTION,
                    "input": llm_text,
                    "output": LABEL_TEXT[label],
                    "text": llm_text,
                    "label": label,
                }
            )

        chain_idx += 1

    os.makedirs(pt_dir, exist_ok=True)
    torch.save(
        {
            "dense_feat": torch.from_numpy(dense_out),
            "sparse_feat": torch.from_numpy(sparse_out),
            "labels": torch.from_numpy(labels_out),
            "valid_len": torch.from_numpy(vlen_out),
            "delays": torch.from_numpy(delays_out),
            "dense_names": DENSE_COLS,
            "sparse_names": SPARSE_COLS,
            "dense_names_cn": [DENSE_NAME_MAP_CN[name] for name in DENSE_COLS],
            "sparse_names_cn": [SPARSE_NAME_MAP_CN[name] for name in SPARSE_COLS],
        },
        pt_path,
    )

    os.makedirs(llm_dir, exist_ok=True)
    with open(llm_path, "w", encoding="utf-8") as f:
        for item in llm_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {"file": csv_path, "chains": int(num_chains), "samples": len(llm_records), "skipped": False}


def process_month_files(month_files, encoders, overwrite=False):
    if not month_files:
        return []
    worker_count = min(MAX_WORKERS, len(month_files))
    results = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_worker,
        initargs=(encoders,),
    ) as executor:
        futures = [executor.submit(process_day_file, path, overwrite) for path in month_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="    Days", leave=False):
            results.append(future.result())
    return results


def group_files_by_month(year_files):
    grouped = {}
    for path in year_files:
        month = os.path.basename(os.path.dirname(path))
        grouped.setdefault(month, []).append(path)
    return grouped


def process_year(year, overwrite=False):
    year_files = collect_year_files(year)
    if not year_files:
        print(f"[SKIP] {year}: no files")
        return

    print(f"Found {len(year_files)} daily files for {year}")
    encoders = build_year_encoders(year_files)
    save_reverse_maps(year, encoders)

    month_groups = group_files_by_month(year_files)
    total_written = 0
    total_skipped = 0
    total_chains = 0
    total_samples = 0

    for month in sorted(month_groups):
        month_files = month_groups[month]
        print(f"Processing {year}-{month} ({len(month_files)} files)...")
        results = process_month_files(month_files, encoders, overwrite=overwrite)

        month_written = sum(1 for item in results if not item["skipped"])
        month_skipped = sum(1 for item in results if item["skipped"])
        month_chains = sum((item["chains"] or 0) for item in results)
        month_samples = sum((item["samples"] or 0) for item in results)
        total_written += month_written
        total_skipped += month_skipped
        total_chains += month_chains
        total_samples += month_samples
        print(
            f"  Month {month} done: written={month_written}, skipped={month_skipped}, "
            f"chains={month_chains:,}, llm_samples={month_samples:,}"
        )

    print(
        f"Done {year}: written={total_written}, skipped={total_skipped}, "
        f"total_chains={total_chains:,}, total_llm_samples={total_samples:,}"
    )


def main():
    overwrite = "--overwrite" in sys.argv
    year_args = [int(arg) for arg in sys.argv[1:] if arg != "--overwrite"]
    years = year_args if year_args else list(range(2016, 2026))
    print("MAX_WORKERS =", MAX_WORKERS)
    for year in years:
        process_year(year, overwrite=overwrite)
    print("\nAll done!")


if __name__ == "__main__":
    main()
