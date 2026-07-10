"""Build enhanced compact LLM JSONL from daily flight-chain samples.

Input:
- scripts/Aeolus_V2/dataset/Flight_Chain_LLM/{year}/{month}/flight_chain_llm_*.jsonl
- scripts/flight_with_weather/{year}/{month}/flight_with_weather_*.csv

Output:
- scripts/Aeolus_V2/dataset/Flight_Chain_LLM_strict/{year}/{month}/flight_chain_llm_strict_*.jsonl
"""

import glob
import json
import math
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "三模态数据库建立说明", "scripts"))
SOURCE_LLM_ROOT = os.path.join(SCRIPT_DIR, "Aeolus_V2", "dataset", "Flight_Chain_LLM")
SOURCE_FLIGHT_ROOT = os.path.join(SCRIPT_DIR, "flight_with_weather")
OUT_ROOT = os.path.join(SCRIPT_DIR, "Aeolus_V2", "dataset", "Flight_Chain_LLM_strict")

YEAR = 2024
WINDOW_MINUTES = 60
ROUTE_HISTORY_N = 5
NEARBY_RADIUS_KM = 200.0
SOURCE_MAX_CHAIN = 6
SCHEMA_VERSION = "chain_llm_strict"
PROMPT_VERSION = "propagation_capsule_strict_v1"
OBSERVATION_POLICY = "strict_actual_event_before_prediction"

TASK_INSTRUCTION = "根据传播胶囊判断当前航班是否会出发延误超过15分钟。只输出：正常 或 延误。"
LABEL_TEXT = {0: "正常", 1: "延误"}

FLIGHT_COLS = [
    "FL_DATE",
    "OP_CARRIER",
    "OP_CARRIER_FL_NUM",
    "TAIL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_DELAY",
    "CRS_ARR_TIME",
    "ARR_DELAY",
    "CRS_ELAPSED_TIME",
    "FLIGHTS",
    "MONTH",
    "DAY_OF_WEEK",
    "O_TEMP",
    "O_PRCP",
    "O_WSPD",
    "D_TEMP",
    "D_PRCP",
    "D_WSPD",
    "O_LATITUDE",
    "O_LONGITUDE",
    "D_LATITUDE",
    "D_LONGITUDE",
]


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


def hhmm_to_minutes(value):
    if pd.isna(value):
        return 0
    hhmm = int(float(value))
    hour = max(0, min(23, hhmm // 100))
    minute = max(0, min(59, hhmm % 100))
    return hour * 60 + minute


def hhmm_to_text(value):
    minutes = hhmm_to_minutes(value)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def fmt(value, digits=1):
    if value is None or pd.isna(value):
        return "缺失"
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.{digits}f}"


def haversine_km(lat1, lon1, lat2, lon2):
    if any(v is None or pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return float("inf")
    r = 6371.0
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def list_daily_llm(year):
    return sorted(glob.glob(os.path.join(SOURCE_LLM_ROOT, str(year), "*", "flight_chain_llm_*.jsonl")))


def flight_csv_for_llm(llm_path):
    month = os.path.basename(os.path.dirname(llm_path))
    day_tag = os.path.basename(llm_path).replace("flight_chain_llm_", "").replace(".jsonl", "")
    yyyy = "20" + day_tag[:2]
    mm = day_tag[2:4]
    dd = day_tag[4:6]
    return os.path.join(SOURCE_FLIGHT_ROOT, yyyy, month, f"flight_with_weather_{yyyy}_{mm}_{dd}.csv")


def output_path_for_llm(llm_path):
    year = os.path.basename(os.path.dirname(os.path.dirname(llm_path)))
    month = os.path.basename(os.path.dirname(llm_path))
    filename = os.path.basename(llm_path).replace("flight_chain_llm_", "flight_chain_llm_strict_")
    out_dir = os.path.join(OUT_ROOT, year, month)
    return os.path.join(out_dir, filename)


def prepare_day_dataframe(csv_path):
    df = pd.read_csv(csv_path, usecols=lambda c: c in FLIGHT_COLS, low_memory=False)
    if df.empty:
        return df

    df = df.copy()
    df["TAIL_NUM_NORM"] = df["TAIL_NUM"].map(normalize_token)
    df["ORIGIN_NORM"] = df["ORIGIN"].map(normalize_token)
    df["DEST_NORM"] = df["DEST"].map(normalize_token)
    df["OP_CARRIER_NORM"] = df["OP_CARRIER"].map(normalize_token)
    df["OP_CARRIER_FL_NUM_NORM"] = df["OP_CARRIER_FL_NUM"].map(normalize_flight_number)
    df["DEP_MIN"] = df["CRS_DEP_TIME"].map(hhmm_to_minutes).astype(np.int16)
    df["ARR_MIN"] = df["CRS_ARR_TIME"].map(hhmm_to_minutes).astype(np.int16)

    for col in [
        "DEP_DELAY",
        "ARR_DELAY",
        "O_TEMP",
        "O_PRCP",
        "O_WSPD",
        "D_TEMP",
        "D_PRCP",
        "D_WSPD",
        "CRS_ELAPSED_TIME",
        "O_LATITUDE",
        "O_LONGITUDE",
        "D_LATITUDE",
        "D_LONGITUDE",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    dep_min = df["DEP_MIN"].astype(np.float32)
    arr_min = df["ARR_MIN"].astype(np.float32)
    arr_sched_min = arr_min + np.where(arr_min < dep_min, 1440.0, 0.0)
    df["ARR_SCHED_MIN"] = arr_sched_min.astype(np.float32)
    df["ACT_DEP_MIN"] = dep_min + df["DEP_DELAY"].astype(np.float32)
    df["ACT_ARR_MIN"] = arr_sched_min + df["ARR_DELAY"].astype(np.float32)

    return df.sort_values(["TAIL_NUM_NORM", "DEP_MIN"], kind="mergesort").reset_index(drop=True)


def build_airport_coords(df, arrays=None):
    if arrays is None:
        arrays = build_day_arrays(df)
    coords = {}
    for idx in range(len(df)):
        origin = arrays["ORIGIN_NORM"][idx]
        dest = arrays["DEST_NORM"][idx]
        if origin and origin not in coords:
            coords[origin] = (safe_float(arrays["O_LATITUDE"][idx]), safe_float(arrays["O_LONGITUDE"][idx]))
        if dest and dest not in coords:
            coords[dest] = (safe_float(arrays["D_LATITUDE"][idx]), safe_float(arrays["D_LONGITUDE"][idx]))
    return coords


def nearby_airports(airport, coords):
    if airport not in coords:
        return []
    lat, lon = coords[airport]
    out = []
    for other, (olat, olon) in coords.items():
        if other == airport:
            continue
        if haversine_km(lat, lon, olat, olon) <= NEARBY_RADIUS_KM:
            out.append(other)
    return out


def build_nearby_cache(coords):
    return {airport: nearby_airports(airport, coords) for airport in coords}


def build_day_arrays(df):
    return {col: df[col].to_numpy(copy=False) for col in df.columns}


def make_group(arrays, idxs):
    idx = np.array(idxs, dtype=np.int64)
    if idx.size == 0:
        return {
            "idx": idx,
            "dep_min": np.array([], dtype=np.int16),
            "arr_sched_min": np.array([], dtype=np.float32),
            "act_dep_min": np.array([], dtype=np.float32),
            "act_arr_min": np.array([], dtype=np.float32),
            "dep_delay": np.array([], dtype=np.float32),
            "arr_delay": np.array([], dtype=np.float32),
        }

    dep_min = arrays["DEP_MIN"][idx].astype(np.int16, copy=False)
    order = np.argsort(dep_min, kind="mergesort")
    idx = idx[order]
    return {
        "idx": idx,
        "dep_min": dep_min[order],
        "arr_sched_min": arrays["ARR_SCHED_MIN"][idx].astype(np.float32, copy=False),
        "act_dep_min": arrays["ACT_DEP_MIN"][idx].astype(np.float32, copy=False),
        "act_arr_min": arrays["ACT_ARR_MIN"][idx].astype(np.float32, copy=False),
        "dep_delay": arrays["DEP_DELAY"][idx].astype(np.float32, copy=False),
        "arr_delay": arrays["ARR_DELAY"][idx].astype(np.float32, copy=False),
    }


def build_indices(df, arrays):
    raw_indices = {
        "route": defaultdict(list),
        "airport_origin": defaultdict(list),
        "airport_dest": defaultdict(list),
        "carrier_origin": defaultdict(list),
    }

    origins = arrays["ORIGIN_NORM"]
    dests = arrays["DEST_NORM"]
    carriers = arrays["OP_CARRIER_NORM"]
    for idx in range(len(df)):
        origin = origins[idx]
        dest = dests[idx]
        carrier = carriers[idx]
        route = (origin, dest)
        raw_indices["route"][route].append(idx)
        raw_indices["airport_origin"][origin].append(idx)
        raw_indices["airport_dest"][dest].append(idx)
        raw_indices["carrier_origin"][(carrier, origin)].append(idx)

    indices = {}
    for name, groups in raw_indices.items():
        indices[name] = {key: make_group(arrays, idxs) for key, idxs in groups.items()}
    return indices


def observed_group(group, current_min, delay_col="DEP_DELAY", limit=None, window=None, exclude_idx=None):
    if group is None or group["idx"].size == 0:
        return None

    if delay_col == "ARR_DELAY":
        sched_min = group["arr_sched_min"]
        event_min = group["act_arr_min"]
        values = group["arr_delay"]
    else:
        sched_min = group["dep_min"].astype(np.float32, copy=False)
        event_min = group["act_dep_min"]
        values = group["dep_delay"]

    mask = sched_min < float(current_min)
    if window is not None:
        mask &= sched_min >= float(current_min - window)
    mask &= np.isfinite(event_min)
    mask &= event_min <= float(current_min)
    if exclude_idx is not None:
        mask &= group["idx"] != int(exclude_idx)

    positions = np.nonzero(mask)[0]
    if positions.size == 0:
        return None

    order = np.argsort(sched_min[positions], kind="mergesort")
    positions = positions[order]
    if limit is not None:
        positions = positions[-limit:]

    return {
        "idx": group["idx"][positions],
        "dep_delay": group["dep_delay"][positions],
        "arr_delay": group["arr_delay"][positions],
    }


def combine_groups(groups):
    valid = [group for group in groups if group is not None and group["idx"].size > 0]
    if not valid:
        return None

    idx = np.concatenate([group["idx"] for group in valid])
    dep_delay = np.concatenate([group["dep_delay"] for group in valid])
    arr_delay = np.concatenate([group["arr_delay"] for group in valid])
    _, unique_pos = np.unique(idx, return_index=True)
    unique_pos = np.sort(unique_pos)
    return {
        "idx": idx[unique_pos],
        "dep_delay": dep_delay[unique_pos],
        "arr_delay": arr_delay[unique_pos],
    }


def summarize_values(values):
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {
            "count": 0,
            "delay_rate": None,
            "mean_delay": None,
            "max_delay": None,
        }

    return {
        "count": int(values.size),
        "delay_rate": float(np.mean(values > 15.0)),
        "mean_delay": float(np.mean(values)),
        "max_delay": float(np.max(values)),
    }


def summarize_group(group, delay_col="DEP_DELAY"):
    if group is None:
        values = np.array([], dtype=np.float32)
    elif delay_col == "ARR_DELAY":
        values = group["arr_delay"]
    else:
        values = group["dep_delay"]
    return summarize_values(values)


def nz(value):
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def clip01(value):
    return float(np.clip(nz(value), 0.0, 1.0))


def build_sample_row_map(df, arrays):
    row_map = {}
    tail_groups = defaultdict(list)
    tails = arrays["TAIL_NUM_NORM"]
    dep_mins = arrays["DEP_MIN"]
    for idx, tail in enumerate(tails):
        if tail:
            tail_groups[tail].append(idx)

    chain_idx = 0
    for idxs in tail_groups.values():
        if len(idxs) < 2:
            continue
        idxs = sorted(idxs, key=lambda idx: int(dep_mins[idx]))
        take = min(len(idxs), SOURCE_MAX_CHAIN)
        for pos in range(take):
            row_map[(chain_idx, pos + 1)] = idxs[pos]
        chain_idx += 1
    return row_map


def lookup_current_row(row_map, item):
    sample_id = item["sample_id"]
    chain_position = int(item.get("chain_position", 1))
    chain_idx = int(sample_id.split("_")[1])
    key = (chain_idx, chain_position)
    if key not in row_map:
        raise IndexError(f"Cannot map sample {sample_id}")
    return row_map[key]


def propagation_from_text(item):
    text = item.get("input") or item.get("text") or ""
    values = {}
    for line in text.splitlines():
        if "：" not in line:
            continue
        key, value = line.split("：", 1)
        values[key.strip()] = value.strip()
    return values


def as_number(text):
    try:
        if text in (None, "", "缺失"):
            return None
        return float(text)
    except Exception:
        return None


def cached_group_summary(cache, name, key, group, current_min, delay_col, limit=None, window=None, exclude_idx=None):
    cache_key = (name, key, int(current_min), limit, window, delay_col, exclude_idx)
    if cache_key not in cache:
        selected = observed_group(
            group,
            current_min,
            delay_col=delay_col,
            limit=limit,
            window=window,
            exclude_idx=exclude_idx,
        )
        cache[cache_key] = summarize_group(selected, delay_col)
    return cache[cache_key]


def cached_nearby_summary(cache, indices, nearby_cache, airport, current_min, index_name, delay_col, exclude_idx=None):
    cache_key = ("nearby", index_name, airport, int(current_min), delay_col, exclude_idx)
    if cache_key not in cache:
        combined = combine_groups([
            observed_group(
                indices[index_name].get(other),
                current_min,
                delay_col=delay_col,
                window=WINDOW_MINUTES,
                exclude_idx=exclude_idx,
            )
            for other in nearby_cache.get(airport, [])
        ])
        cache[cache_key] = summarize_group(combined, delay_col)
    return cache[cache_key]


def build_context(arrays, indices, nearby_cache, context_cache, row_idx, item):
    origin = arrays["ORIGIN_NORM"][row_idx]
    dest = arrays["DEST_NORM"][row_idx]
    carrier = arrays["OP_CARRIER_NORM"][row_idx]
    dep_min = int(arrays["DEP_MIN"][row_idx])
    route = (origin, dest)
    reverse_route = (dest, origin)

    near_origin = nearby_cache.get(origin, [])
    near_dest = nearby_cache.get(dest, [])

    old_summary = propagation_from_text(item)
    prev_arr = as_number(old_summary.get("前序到达延误分钟数"))
    prev_dep = as_number(old_summary.get("前序出发延误分钟数"))
    slack = as_number(old_summary.get("计划过站缓冲时间(分钟)"))
    dep_trend = as_number(old_summary.get("前两段出发延误变化趋势"))

    remaining_slack = None
    propagation_pressure = None
    slack_breached = None
    if slack is not None and prev_arr is not None:
        remaining_slack = slack - max(0.0, prev_arr)
        propagation_pressure = max(0.0, prev_arr) / max(slack, 1.0)
        slack_breached = remaining_slack < 0

    route_context = cached_group_summary(
        context_cache,
        "route",
        route,
        indices["route"].get(route),
        dep_min,
        "DEP_DELAY",
        limit=ROUTE_HISTORY_N,
        exclude_idx=row_idx,
    )
    reverse_route_context = cached_group_summary(
        context_cache,
        "reverse_route",
        reverse_route,
        indices["route"].get(reverse_route),
        dep_min,
        "DEP_DELAY",
        limit=ROUTE_HISTORY_N,
        exclude_idx=row_idx,
    )
    origin_context = cached_group_summary(
        context_cache,
        "origin",
        origin,
        indices["airport_origin"].get(origin),
        dep_min,
        "DEP_DELAY",
        window=WINDOW_MINUTES,
        exclude_idx=row_idx,
    )
    dest_context = cached_group_summary(
        context_cache,
        "dest",
        dest,
        indices["airport_dest"].get(dest),
        dep_min,
        "ARR_DELAY",
        window=WINDOW_MINUTES,
        exclude_idx=row_idx,
    )
    carrier_origin_context = cached_group_summary(
        context_cache,
        "carrier_origin",
        (carrier, origin),
        indices["carrier_origin"].get((carrier, origin)),
        dep_min,
        "DEP_DELAY",
        window=WINDOW_MINUTES,
        exclude_idx=row_idx,
    )
    near_origin_context = cached_nearby_summary(
        context_cache, indices, nearby_cache, origin, dep_min, "airport_origin", "DEP_DELAY", exclude_idx=row_idx
    )
    near_dest_context = cached_nearby_summary(
        context_cache, indices, nearby_cache, dest, dep_min, "airport_dest", "ARR_DELAY", exclude_idx=row_idx
    )

    chain_pressure = clip01(propagation_pressure)
    airport_pressure = (
        0.45 * clip01(origin_context["delay_rate"])
        + 0.25 * clip01(carrier_origin_context["delay_rate"])
        + 0.15 * clip01(dest_context["delay_rate"])
        + 0.10 * clip01(near_origin_context["delay_rate"])
        + 0.05 * clip01(near_dest_context["delay_rate"])
    )
    route_pressure = (
        0.75 * clip01(route_context["delay_rate"])
        + 0.25 * clip01(reverse_route_context["delay_rate"])
    )
    composite_risk = (
        0.50 * chain_pressure
        + 0.30 * airport_pressure
        + 0.20 * route_pressure
    )

    return {
        "observation_policy": OBSERVATION_POLICY,
        "current": {
            "month": int(arrays["MONTH"][row_idx]) if not pd.isna(arrays["MONTH"][row_idx]) else None,
            "day_of_week": int(arrays["DAY_OF_WEEK"][row_idx]) if not pd.isna(arrays["DAY_OF_WEEK"][row_idx]) else None,
            "dep_time": hhmm_to_text(arrays["CRS_DEP_TIME"][row_idx]),
            "arr_time": hhmm_to_text(arrays["CRS_ARR_TIME"][row_idx]),
            "origin": origin,
            "dest": dest,
            "carrier": carrier,
            "flight_number": arrays["OP_CARRIER_FL_NUM_NORM"][row_idx],
            "tail": arrays["TAIL_NUM_NORM"][row_idx],
            "elapsed_min": safe_float(arrays["CRS_ELAPSED_TIME"][row_idx]),
            "origin_weather": {
                "temp": safe_float(arrays["O_TEMP"][row_idx]),
                "prcp": safe_float(arrays["O_PRCP"][row_idx]),
                "wspd": safe_float(arrays["O_WSPD"][row_idx]),
            },
            "dest_weather": {
                "temp": safe_float(arrays["D_TEMP"][row_idx]),
                "prcp": safe_float(arrays["D_PRCP"][row_idx]),
                "wspd": safe_float(arrays["D_WSPD"][row_idx]),
            },
        },
        "tail_chain": {
            "prev_dep_delay": prev_dep,
            "prev_arr_delay": prev_arr,
            "turnaround_slack_min": slack,
            "remaining_slack_min": remaining_slack,
            "slack_breached": slack_breached,
            "dep_delay_trend": dep_trend,
            "propagation_pressure": propagation_pressure,
        },
        "route_context": route_context,
        "reverse_route_context": reverse_route_context,
        "airport_context": {
            "origin_60m": origin_context,
            "dest_60m": dest_context,
            "carrier_origin_60m": carrier_origin_context,
        },
        "nearby_airport_context": {
            "radius_km": NEARBY_RADIUS_KM,
            "near_origin_airports": near_origin[:8],
            "near_dest_airports": near_dest[:8],
            "near_origin_60m": near_origin_context,
            "near_dest_60m": near_dest_context,
        },
        "propagation_risk": {
            "chain_pressure": chain_pressure,
            "airport_pressure": airport_pressure,
            "route_pressure": route_pressure,
            "composite_risk": composite_risk,
        },
    }


def render_compact_text(features):
    cur = features["current"]
    tail = features["tail_chain"]
    route = features["route_context"]
    airport = features["airport_context"]
    nearby = features["nearby_airport_context"]
    risk = features["propagation_risk"]

    return "\n".join(
        [
            (
                f"当前: {cur['origin']}->{cur['dest']}; {cur['dep_time']}起飞; {cur['arr_time']}到达; "
                f"{cur['carrier']}{cur['flight_number']}; 尾号{cur['tail']}; 飞行{fmt(cur['elapsed_min'])}分钟"
            ),
            (
                f"天气: 出发温度{fmt(cur['origin_weather']['temp'])}; 出发降水{fmt(cur['origin_weather']['prcp'])}; "
                f"出发风速{fmt(cur['origin_weather']['wspd'])}; 到达风速{fmt(cur['dest_weather']['wspd'])}"
            ),
            (
                f"同机传播: 前序出发延误{fmt(tail['prev_dep_delay'])}; 前序到达延误{fmt(tail['prev_arr_delay'])}; "
                f"过站缓冲{fmt(tail['turnaround_slack_min'])}; 剩余缓冲{fmt(tail['remaining_slack_min'])}; "
                f"传播压力{fmt(tail['propagation_pressure'], 2)}; 趋势{fmt(tail['dep_delay_trend'])}"
            ),
            (
                f"同航段近{ROUTE_HISTORY_N}班: 数量{route['count']}; 延误率{fmt(route['delay_rate'], 2)}; "
                f"均值{fmt(route['mean_delay'])}; 最大{fmt(route['max_delay'])}"
            ),
            (
                f"机场近{WINDOW_MINUTES}分钟: 出发机场数量{airport['origin_60m']['count']}; "
                f"出发延误率{fmt(airport['origin_60m']['delay_rate'], 2)}; "
                f"出发均值{fmt(airport['origin_60m']['mean_delay'])}; "
                f"同承运人延误率{fmt(airport['carrier_origin_60m']['delay_rate'], 2)}"
            ),
            (
                f"邻近机场{int(NEARBY_RADIUS_KM)}km: 出发邻近数{nearby['near_origin_60m']['count']}; "
                f"出发邻近延误率{fmt(nearby['near_origin_60m']['delay_rate'], 2)}; "
                f"到达邻近延误率{fmt(nearby['near_dest_60m']['delay_rate'], 2)}"
            ),
            (
                f"综合风险: 链式{fmt(risk['chain_pressure'], 2)}; 机场{fmt(risk['airport_pressure'], 2)}; "
                f"航段{fmt(risk['route_pressure'], 2)}; 合计{fmt(risk['composite_risk'], 2)}"
            ),
        ]
    )


def enhance_day(llm_path, overwrite=False):
    out_path = output_path_for_llm(llm_path)
    if os.path.exists(out_path) and not overwrite:
        return {"path": out_path, "rows": None, "skipped": True}

    csv_path = flight_csv_for_llm(llm_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing flight csv for {llm_path}: {csv_path}")

    df = prepare_day_dataframe(csv_path)
    arrays = build_day_arrays(df)
    indices = build_indices(df, arrays)
    coords = build_airport_coords(df, arrays)
    nearby_cache = build_nearby_cache(coords)
    row_map = build_sample_row_map(df, arrays)
    context_cache = {}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = 0
    with open(llm_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            row_idx = lookup_current_row(row_map, item)
            features = build_context(arrays, indices, nearby_cache, context_cache, row_idx, item)
            compact_text = render_compact_text(features)

            out = {
                **item,
                "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "observation_policy": OBSERVATION_POLICY,
                "instruction": TASK_INSTRUCTION,
                "features": features,
                "compact_text": compact_text,
                "input": compact_text,
                "text": compact_text,
                "output": LABEL_TEXT[int(item["label"])],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows += 1

    return {"path": out_path, "rows": rows, "skipped": False}


def enhance_day_job(args):
    return enhance_day(*args)


def main():
    parser = argparse.ArgumentParser(description="Build enhanced compact LLM daily JSONL files.")
    parser.add_argument("years", nargs="*", type=int, default=[YEAR])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--date", help="Only process one yymmdd date, for example 240101.")
    parser.add_argument("--max-days", type=int, help="Only process the first N daily files after filtering.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel daily workers. Use 1 to disable multiprocessing.")
    args = parser.parse_args()

    for year in args.years:
        files = list_daily_llm(year)
        if args.date:
            files = [path for path in files if os.path.basename(path) == f"flight_chain_llm_{args.date}.jsonl"]
        if args.max_days is not None:
            files = files[:args.max_days]
        if not files:
            raise FileNotFoundError(f"No source daily LLM files found for {year}: {SOURCE_LLM_ROOT}")

        total_rows = 0
        skipped = 0
        print(f"Enhancing {year}: {len(files)} daily files")
        workers = max(1, int(args.workers))
        if workers == 1 or len(files) == 1:
            iterator = (enhance_day(path, overwrite=args.overwrite) for path in files)
            for result in tqdm(iterator, total=len(files), desc=f"LLM strict {year}"):
                if result["skipped"]:
                    skipped += 1
                else:
                    total_rows += result["rows"]
        else:
            jobs = [(path, args.overwrite) for path in files]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(enhance_day_job, job) for job in jobs]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"LLM strict {year}"):
                    result = future.result()
                    if result["skipped"]:
                        skipped += 1
                    else:
                        total_rows += result["rows"]

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "observation_policy": OBSERVATION_POLICY,
            "source_root": SOURCE_LLM_ROOT,
            "flight_root": SOURCE_FLIGHT_ROOT,
            "output_root": OUT_ROOT,
            "year": year,
            "daily_files": len(files),
            "written_rows": total_rows,
            "skipped_files": skipped,
            "window_minutes": WINDOW_MINUTES,
            "route_history_n": ROUTE_HISTORY_N,
            "nearby_radius_km": NEARBY_RADIUS_KM,
            "workers": workers,
        }
        os.makedirs(OUT_ROOT, exist_ok=True)
        with open(os.path.join(OUT_ROOT, f"{year}_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
