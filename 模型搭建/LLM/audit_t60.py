"""Audit the T-60 event boundary and chronological dataset split."""
import argparse
import importlib.util
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data_t60")
VARIANT_FILES = {
    "probe": {"train": "train_probe.jsonl", "val": "val_probe.jsonl", "test": "test_probe.jsonl"},
    "medium": {"train": "train_10k.jsonl", "val": "val_2k.jsonl", "test": "test_final.jsonl"},
    "full": {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"},
}


def load_preprocessor():
    path = os.path.join(HERE, "0_数据整合.py")
    spec = importlib.util.spec_from_file_location("t60_preprocessor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_window_function():
    module = load_preprocessor()
    group = {
        "idx": np.arange(4, dtype=np.int64),
        "dep_min": np.array([480, 540, 590, 610], dtype=np.int16),
        "dep_sched_utc": np.array([480, 540, 590, 610], dtype=np.float32),
        "arr_sched_utc": np.array([480, 540, 590, 610], dtype=np.float32),
        "act_dep_utc": np.array([500, 550, 600, 601], dtype=np.float32),
        "act_arr_utc": np.array([500, 550, 600, 601], dtype=np.float32),
        "dep_component": np.zeros(4, dtype=np.int16),
        "arr_component": np.zeros(4, dtype=np.int16),
        "dep_delay": np.array([20, 10, 10, -9], dtype=np.float32),
        "arr_delay": np.array([20, 10, 10, -9], dtype=np.float32),
    }
    selected = module.observed_group(group, cutoff_min=600, component=0, window=60)
    selected_ids = selected["idx"].tolist()
    if selected_ids != [1, 2]:
        raise AssertionError(f"Expected events [1, 2] inside [540, 600], got {selected_ids}")
    print("Event-window boundary: PASS (future and pre-window events excluded)")


def chain_key(item):
    parts = item["sample_id"].split("_")
    return "_".join(parts[:2])


def read_identity_sets(path):
    dates, samples, chains = set(), set(), set()
    rows = positives = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            dates.add(item["date"])
            samples.add(item["sample_id"])
            chains.add((item["date"], chain_key(item)))
            positives += int(item["label"])
            rows += 1
            current = item.get("features", {}).get("current", {})
            if current.get("prediction_horizon_min") != 60:
                raise AssertionError(f"Non-T60 sample found: {item['sample_id']}")
    if not rows:
        raise AssertionError(f"Empty split: {path}")
    return {
        "dates": dates,
        "samples": samples,
        "chains": chains,
        "rows": rows,
        "positive_rate": positives / rows,
    }


def audit_split(variant):
    stats = {}
    for split, filename in VARIANT_FILES[variant].items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Build the {variant} split first: {path}")
        stats[split] = read_identity_sets(path)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if stats[left]["dates"] & stats[right]["dates"]:
            raise AssertionError(f"Date overlap between {left} and {right}")
        if stats[left]["samples"] & stats[right]["samples"]:
            raise AssertionError(f"Sample overlap between {left} and {right}")
        if stats[left]["chains"] & stats[right]["chains"]:
            raise AssertionError(f"Chain overlap between {left} and {right}")

    if not max(stats["train"]["dates"]) < min(stats["val"]["dates"]):
        raise AssertionError("Training dates are not strictly before validation dates")
    if not max(stats["val"]["dates"]) < min(stats["test"]["dates"]):
        raise AssertionError("Validation dates are not strictly before test dates")

    for split in ("train", "val", "test"):
        item = stats[split]
        print(
            f"{split}: rows={item['rows']}, dates={min(item['dates'])}..{max(item['dates'])}, "
            f"positive_rate={item['positive_rate']:.4f}"
        )
    print(f"Chronological {variant} split: PASS (no date/sample/chain overlap)")


def main():
    parser = argparse.ArgumentParser(description="Audit T-60 preprocessing and split isolation.")
    parser.add_argument("--variant", choices=VARIANT_FILES, default="probe")
    parser.add_argument("--window-only", action="store_true")
    args = parser.parse_args()
    audit_window_function()
    if not args.window_only:
        audit_split(args.variant)


if __name__ == "__main__":
    main()
