"""Build enhanced compact LLM JSONL from daily flight-chain samples.

Input:
- scripts/Aeolus_V2/dataset/Flight_Chain_LLM/{year}/{month}/flight_chain_llm_*.jsonl
- scripts/flight_with_weather/{year}/{month}/flight_with_weather_*.csv

Output:
- scripts/Aeolus_V2/dataset/Flight_Chain_LLM_t60/{year}/{month}/flight_chain_llm_t60_*.jsonl
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
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from service.prompt import render_compact_features


SCRIPT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "三模态数据库建立说明", "scripts"))
SOURCE_LLM_ROOT = os.path.join(SCRIPT_DIR, "Aeolus_V2", "dataset", "Flight_Chain_LLM")
SOURCE_FLIGHT_ROOT = os.path.join(SCRIPT_DIR, "flight_with_weather")
OUT_ROOT = os.path.join(SCRIPT_DIR, "Aeolus_V2", "dataset", "Flight_Chain_LLM_t60")

YEAR = 2024
PREDICTION_HORIZON_MINUTES = 60
WINDOW_MINUTES = 60
ROUTE_HISTORY_N = 5
NEARBY_RADIUS_KM = 200.0
SOURCE_MAX_CHAIN = 6
SCHEMA_VERSION = "chain_llm_t60"
PROMPT_VERSION = "propagation_capsule_t60_operational"
OBSERVATION_POLICY = "utc_actual_event_at_or_before_t_minus_60"

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


def minute_offset_to_text(minutes):
    day_offset, minute_of_day = divmod(int(minutes), 1440)
    clock = f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"
    return clock if day_offset == 0 else f"D{day_offset:+d} {clock}"


def normalize_clock_delta(minutes):
    """Normalize a local-clock difference to the nearest same-day UTC offset."""
    return float((float(minutes) + 720.0) % 1440.0 - 720.0)


def build_airport_clock_map(df):
    """Infer per-airport relative UTC offsets from scheduled elapsed times."""
    pair_deltas = defaultdict(list)
    for row in df.itertuples(index=False):
        origin = normalize_token(row.ORIGIN)
        dest = normalize_token(row.DEST)
        elapsed = safe_float(row.CRS_ELAPSED_TIME)
        if not origin or not dest or elapsed is None:
            continue
        delta = normalize_clock_delta(hhmm_to_minutes(row.CRS_ARR_TIME) - hhmm_to_minutes(row.CRS_DEP_TIME) - elapsed)
        if origin <= dest:
            pair_deltas[(origin, dest)].append(delta)
        else:
            pair_deltas[(dest, origin)].append(-delta)

    airports = sorted((set(df["ORIGIN_NORM"]) | set(df["DEST_NORM"])) - {""})
    airport_index = {airport: idx for idx, airport in enumerate(airports)}
    parent = list(range(len(airports)))

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    rows, cols, values, targets = [], [], [], []
    equation = 0
    for (origin, dest), deltas in pair_deltas.items():
        left, right = airport_index[origin], airport_index[dest]
        union(left, right)
        weight = math.sqrt(len(deltas))
        rows.extend([equation, equation])
        cols.extend([left, right])
        values.extend([-weight, weight])
        targets.append(float(np.median(deltas)) * weight)
        equation += 1

    roots = {}
    for idx in range(len(airports)):
        roots.setdefault(find(idx), idx)
    for anchor in roots.values():
        rows.append(equation)
        cols.append(anchor)
        values.append(1000.0)
        targets.append(0.0)
        equation += 1

    if airports:
        matrix = coo_matrix((values, (rows, cols)), shape=(equation, len(airports))).tocsr()
        solution = lsqr(matrix, np.asarray(targets, dtype=np.float64), atol=1e-8, btol=1e-8)[0]
        solution = np.round(solution / 30.0) * 30.0
    else:
        solution = np.array([], dtype=np.float64)

    root_ids = {root: component_id for component_id, root in enumerate(sorted(roots))}
    offsets = {airport: float(solution[idx]) for airport, idx in airport_index.items()}
    components = {airport: root_ids[find(idx)] for airport, idx in airport_index.items()}
    return offsets, components


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
    filename = os.path.basename(llm_path).replace("flight_chain_llm_", "flight_chain_llm_t60_")
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

    offsets, components = build_airport_clock_map(df)
    df["ORIGIN_CLOCK_OFFSET"] = df["ORIGIN_NORM"].map(offsets).fillna(0.0).astype(np.float32)
    df["DEST_CLOCK_OFFSET"] = df["DEST_NORM"].map(offsets).fillna(0.0).astype(np.float32)
    df["ORIGIN_TIME_COMPONENT"] = df["ORIGIN_NORM"].map(components).fillna(-1).astype(np.int16)
    df["DEST_TIME_COMPONENT"] = df["DEST_NORM"].map(components).fillna(-1).astype(np.int16)

    dep_sched_utc = df["DEP_MIN"].astype(np.float32) - df["ORIGIN_CLOCK_OFFSET"]
    arr_sched_utc = dep_sched_utc + df["CRS_ELAPSED_TIME"].astype(np.float32)
    df["DEP_SCHED_UTC"] = dep_sched_utc
    df["ARR_SCHED_UTC"] = arr_sched_utc
    df["ACT_DEP_UTC"] = dep_sched_utc + df["DEP_DELAY"].astype(np.float32)
    df["ACT_ARR_UTC"] = arr_sched_utc + df["ARR_DELAY"].astype(np.float32)

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
            "dep_sched_utc": np.array([], dtype=np.float32),
            "arr_sched_utc": np.array([], dtype=np.float32),
            "act_dep_utc": np.array([], dtype=np.float32),
            "act_arr_utc": np.array([], dtype=np.float32),
            "dep_component": np.array([], dtype=np.int16),
            "arr_component": np.array([], dtype=np.int16),
            "dep_delay": np.array([], dtype=np.float32),
            "arr_delay": np.array([], dtype=np.float32),
        }

    dep_min = arrays["DEP_MIN"][idx].astype(np.int16, copy=False)
    dep_sched_utc = arrays["DEP_SCHED_UTC"][idx].astype(np.float32, copy=False)
    order = np.argsort(dep_sched_utc, kind="mergesort")
    idx = idx[order]
    return {
        "idx": idx,
        "dep_min": dep_min[order],
        "dep_sched_utc": arrays["DEP_SCHED_UTC"][idx].astype(np.float32, copy=False),
        "arr_sched_utc": arrays["ARR_SCHED_UTC"][idx].astype(np.float32, copy=False),
        "act_dep_utc": arrays["ACT_DEP_UTC"][idx].astype(np.float32, copy=False),
        "act_arr_utc": arrays["ACT_ARR_UTC"][idx].astype(np.float32, copy=False),
        "dep_component": arrays["ORIGIN_TIME_COMPONENT"][idx].astype(np.int16, copy=False),
        "arr_component": arrays["DEST_TIME_COMPONENT"][idx].astype(np.int16, copy=False),
        "dep_delay": arrays["DEP_DELAY"][idx].astype(np.float32, copy=False),
        "arr_delay": arrays["ARR_DELAY"][idx].astype(np.float32, copy=False),
    }


def build_indices(df, arrays):
    raw_indices = {
        "tail": defaultdict(list),
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
        tail = arrays["TAIL_NUM_NORM"][idx]
        route = (origin, dest)
        raw_indices["tail"][tail].append(idx)
        raw_indices["route"][route].append(idx)
        raw_indices["airport_origin"][origin].append(idx)
        raw_indices["airport_dest"][dest].append(idx)
        raw_indices["carrier_origin"][(carrier, origin)].append(idx)

    indices = {}
    for name, groups in raw_indices.items():
        indices[name] = {key: make_group(arrays, idxs) for key, idxs in groups.items()}
    return indices


def observed_group(group, cutoff_min, component, delay_col="DEP_DELAY", limit=None, window=None, exclude_idx=None):
    if group is None or group["idx"].size == 0:
        return None

    if delay_col == "ARR_DELAY":
        event_min = group["act_arr_utc"]
        event_component = group["arr_component"]
    else:
        event_min = group["act_dep_utc"]
        event_component = group["dep_component"]

    mask = np.ones(group["idx"].shape, dtype=bool)
    mask &= event_component == int(component)
    mask &= np.isfinite(event_min)
    mask &= event_min <= float(cutoff_min)
    if window is not None:
        mask &= event_min >= float(cutoff_min - window)
    if exclude_idx is not None:
        mask &= group["idx"] != int(exclude_idx)

    positions = np.nonzero(mask)[0]
    if positions.size == 0:
        return None

    order = np.argsort(event_min[positions], kind="mergesort")
    positions = positions[order]
    if limit is not None:
        positions = positions[-limit:]

    selected_event_min = event_min[positions]
    if np.any(selected_event_min > float(cutoff_min)):
        raise RuntimeError("Post-cutoff event entered an observed context")
    if window is not None and np.any(selected_event_min < float(cutoff_min - window)):
        raise RuntimeError("Event outside the lookback window entered an observed context")

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


def strict_tail_context(arrays, indices, row_idx, cutoff_min, component):
    """Return previous-tail outcomes known at the fixed prediction cutoff."""
    tail = arrays["TAIL_NUM_NORM"][row_idx]
    group = indices["tail"].get(tail)
    previous = [] if group is None else [
        int(idx)
        for idx, sched, comp in zip(group["idx"], group["dep_sched_utc"], group["dep_component"])
        if int(idx) != row_idx and int(comp) == int(component) and float(sched) < float(arrays["DEP_SCHED_UTC"][row_idx])
    ]

    if not previous:
        return {
            "has_previous": False,
            "prev_departed": None,
            "prev_arrived": None,
            "prev_dep_delay": None,
            "prev_arr_delay": None,
            "turnaround_slack_min": None,
            "dep_delay_trend": None,
        }

    prev_idx = previous[-1]
    prev_departed = bool(
        np.isfinite(arrays["ACT_DEP_UTC"][prev_idx])
        and arrays["ACT_DEP_UTC"][prev_idx] <= cutoff_min
    )
    prev_arrived = bool(
        np.isfinite(arrays["ACT_ARR_UTC"][prev_idx])
        and arrays["ACT_ARR_UTC"][prev_idx] <= cutoff_min
    )
    prev_dep = safe_float(arrays["DEP_DELAY"][prev_idx]) if prev_departed else None
    prev_arr = safe_float(arrays["ARR_DELAY"][prev_idx]) if prev_arrived else None
    slack = float(arrays["DEP_SCHED_UTC"][row_idx]) - float(arrays["ARR_SCHED_UTC"][prev_idx])

    observed_dep = [
        safe_float(arrays["DEP_DELAY"][idx])
        for idx in previous
        if np.isfinite(arrays["ACT_DEP_UTC"][idx])
        and arrays["ACT_DEP_UTC"][idx] <= cutoff_min
        and np.isfinite(arrays["DEP_DELAY"][idx])
    ]
    dep_trend = observed_dep[-1] - observed_dep[-2] if len(observed_dep) >= 2 else None

    return {
        "has_previous": True,
        "prev_departed": prev_departed,
        "prev_arrived": prev_arrived,
        "prev_dep_delay": prev_dep,
        "prev_arr_delay": prev_arr,
        "turnaround_slack_min": slack,
        "dep_delay_trend": dep_trend,
    }


def cached_group_summary(cache, name, key, group, cutoff_min, component, delay_col, limit=None, window=None, exclude_idx=None):
    cache_key = (name, key, int(cutoff_min), int(component), limit, window, delay_col, exclude_idx)
    if cache_key not in cache:
        selected = observed_group(
            group,
            cutoff_min,
            component,
            delay_col=delay_col,
            limit=limit,
            window=window,
            exclude_idx=exclude_idx,
        )
        cache[cache_key] = summarize_group(selected, delay_col)
    return cache[cache_key]


def cached_nearby_summary(cache, indices, nearby_cache, airport, cutoff_min, component, index_name, delay_col, exclude_idx=None):
    cache_key = ("nearby", index_name, airport, int(cutoff_min), int(component), delay_col, exclude_idx)
    if cache_key not in cache:
        combined = combine_groups([
            observed_group(
                indices[index_name].get(other),
                cutoff_min,
                component,
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
    cutoff_min = float(arrays["DEP_SCHED_UTC"][row_idx]) - PREDICTION_HORIZON_MINUTES
    component = int(arrays["ORIGIN_TIME_COMPONENT"][row_idx])
    route = (origin, dest)
    reverse_route = (dest, origin)

    tail_context = strict_tail_context(arrays, indices, row_idx, cutoff_min, component)

    route_context = cached_group_summary(
        context_cache,
        "route",
        route,
        indices["route"].get(route),
        cutoff_min,
        component,
        "DEP_DELAY",
        limit=ROUTE_HISTORY_N,
        exclude_idx=row_idx,
    )
    reverse_route_context = cached_group_summary(
        context_cache,
        "reverse_route",
        reverse_route,
        indices["route"].get(reverse_route),
        cutoff_min,
        component,
        "DEP_DELAY",
        limit=ROUTE_HISTORY_N,
        exclude_idx=row_idx,
    )
    origin_context = cached_group_summary(
        context_cache,
        "origin",
        origin,
        indices["airport_origin"].get(origin),
        cutoff_min,
        component,
        "DEP_DELAY",
        window=WINDOW_MINUTES,
        exclude_idx=row_idx,
    )
    dest_context = cached_group_summary(
        context_cache,
        "dest",
        dest,
        indices["airport_dest"].get(dest),
        cutoff_min,
        component,
        "ARR_DELAY",
        window=WINDOW_MINUTES,
        exclude_idx=row_idx,
    )
    carrier_origin_context = cached_group_summary(
        context_cache,
        "carrier_origin",
        (carrier, origin),
        indices["carrier_origin"].get((carrier, origin)),
        cutoff_min,
        component,
        "DEP_DELAY",
        window=WINDOW_MINUTES,
        exclude_idx=row_idx,
    )
    near_origin_context = cached_nearby_summary(
        context_cache, indices, nearby_cache, origin, cutoff_min, component, "airport_origin", "DEP_DELAY", exclude_idx=row_idx
    )
    near_dest_context = cached_nearby_summary(
        context_cache, indices, nearby_cache, dest, cutoff_min, component, "airport_dest", "ARR_DELAY", exclude_idx=row_idx
    )

    return {
        "current": {
            "month": int(arrays["MONTH"][row_idx]) if not pd.isna(arrays["MONTH"][row_idx]) else None,
            "day_of_week": int(arrays["DAY_OF_WEEK"][row_idx]) if not pd.isna(arrays["DAY_OF_WEEK"][row_idx]) else None,
            "dep_time": hhmm_to_text(arrays["CRS_DEP_TIME"][row_idx]),
            "observation_time": minute_offset_to_text(dep_min - PREDICTION_HORIZON_MINUTES),
            "prediction_horizon_min": PREDICTION_HORIZON_MINUTES,
            "origin": origin,
            "dest": dest,
            "carrier": carrier,
            "elapsed_min": safe_float(arrays["CRS_ELAPSED_TIME"][row_idx]),
        },
        "tail_chain": tail_context,
        "route_context": {
            "count": route_context["count"],
            "delay_rate": route_context["delay_rate"],
        },
        "reverse_route_context": {
            "count": reverse_route_context["count"],
            "delay_rate": reverse_route_context["delay_rate"],
        },
        "airport_context": {
            "origin_60m": {"count": origin_context["count"], "delay_rate": origin_context["delay_rate"]},
            "dest_60m": {"count": dest_context["count"], "delay_rate": dest_context["delay_rate"]},
            "carrier_origin_60m": {
                "count": carrier_origin_context["count"],
                "delay_rate": carrier_origin_context["delay_rate"],
            },
        },
        "nearby_airport_context": {
            "near_origin_60m": {
                "count": near_origin_context["count"],
                "delay_rate": near_origin_context["delay_rate"],
            },
            "near_dest_60m": {
                "count": near_dest_context["count"],
                "delay_rate": near_dest_context["delay_rate"],
            },
        },
    }


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
            compact_text = render_compact_features(
                features,
                chain_position=item.get("chain_position"),
                chain_valid_length=item.get("chain_valid_length"),
            )

            out = {
                "sample_id": item["sample_id"],
                "date": item["date"],
                "chain_position": item["chain_position"],
                "chain_valid_length": item["chain_valid_length"],
                "label": int(item["label"]),
                "features": features,
                "compact_text": compact_text,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            rows += 1

    return {"path": out_path, "rows": rows, "skipped": False}


def enhance_day_job(args):
    llm_path = args[0]
    try:
        return enhance_day(*args)
    except Exception as exc:
        raise RuntimeError(f"Failed T-60 preprocessing for {llm_path}: {exc}") from exc


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
            for result in tqdm(iterator, total=len(files), desc=f"LLM T-60 {year}"):
                if result["skipped"]:
                    skipped += 1
                else:
                    total_rows += result["rows"]
        else:
            jobs = [(path, args.overwrite) for path in files]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(enhance_day_job, job) for job in jobs]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"LLM T-60 {year}"):
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
            "prediction_horizon_minutes": PREDICTION_HORIZON_MINUTES,
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
