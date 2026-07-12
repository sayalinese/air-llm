import argparse
import glob
import hashlib
import json
import math
import os


YEAR = 2024
DAILY_LLM_ROOT = r"C:\Users\16960\Desktop\期末论文\三模态数据库建立说明\scripts\Aeolus_V2\dataset\Flight_Chain_LLM_t60"
OUT_DIR = r"C:\Users\16960\Desktop\期末论文\模型搭建\LLM\data_t60"
FILE_PREFIX = "flight_chain_llm_t60_"
SCHEMA_VERSION = "chain_llm_t60"
PROMPT_VERSION = "propagation_capsule_t60_operational"
OBSERVATION_POLICY = "utc_actual_event_at_or_before_t_minus_60"
PREDICTION_HORIZON_MINUTES = 60
PROBE_ROWS = {"train": 2048, "val": 512, "test": 512}
PROBE_DAYS = {"train": 48, "val": 16, "test": 16}
MEDIUM_ROWS = {"train": 10000, "val": 2000, "test": 5000}
MEDIUM_DAYS = {"train": 96, "val": 32, "test": 32}
PROBE_SEED = 42


def list_daily_jsonl():
    files = sorted(glob.glob(os.path.join(DAILY_LLM_ROOT, str(YEAR), "*", f"{FILE_PREFIX}*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No daily LLM jsonl found under {DAILY_LLM_ROOT}\\{YEAR}")
    return files


def date_from_path(path):
    return os.path.basename(path).replace(FILE_PREFIX, "").replace(".jsonl", "")


def split_files(files):
    if len(files) < 5:
        raise ValueError(f"Need full enhanced daily files before splitting; found {len(files)}")
    c1, c2 = int(len(files) * 0.6), int(len(files) * 0.8)
    return {"train": files[:c1], "val": files[c1:c2], "test": files[c2:]}


def merge_split(split_name, files):
    out_path = os.path.join(OUT_DIR, f"{split_name}.jsonl")
    count = 0
    positives = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for path in files:
            with open(path, encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
                    positives += int(item["label"])
    return out_path, count, positives


def evenly_spaced(files, count):
    count = min(count, len(files))
    if count == len(files):
        return files
    return [files[round(i * (len(files) - 1) / (count - 1))] for i in range(count)]


def chain_key(item):
    parts = item["sample_id"].split("_")
    return "_".join(parts[:2])


def load_chains(path):
    chains = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            chains.setdefault(chain_key(item), []).append(item)
    return chains


def stable_rank(split_name, day, key):
    value = f"{PROBE_SEED}|{split_name}|{day}|{key}".encode("utf-8")
    return hashlib.sha256(value).digest()


def build_sample(output_name, files, target_rows, target_days, rank_name=None):
    selected_files = evenly_spaced(files, target_days)
    per_day = math.ceil(target_rows / len(selected_files))
    selected = []

    for path in selected_files:
        day = date_from_path(path)
        chains = load_chains(path)
        rank_name = rank_name or output_name
        ranked = sorted(chains.items(), key=lambda pair: stable_rank(rank_name, day, pair[0]))
        day_rows = 0
        for _, rows in ranked:
            selected.extend(rows)
            day_rows += len(rows)
            if day_rows >= per_day:
                break

    selected.sort(key=lambda item: (item["date"], chain_key(item), item["chain_position"]))
    out_path = os.path.join(OUT_DIR, f"{output_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    positives = sum(int(item["label"]) for item in selected)
    return {
        "path": out_path,
        "rows": len(selected),
        "positive_rate": positives / max(1, len(selected)),
        "num_days": len(selected_files),
        "dates": [date_from_path(path) for path in selected_files],
        "whole_chain_sampling": True,
        "seed": PROBE_SEED,
    }


def build_probe(split_name, files):
    return build_sample(
        f"{split_name}_probe",
        files,
        PROBE_ROWS[split_name],
        PROBE_DAYS[split_name],
        rank_name=split_name,
    )


def build_medium(splits, probe_stats):
    train = build_sample("train_10k", splits["train"], MEDIUM_ROWS["train"], MEDIUM_DAYS["train"])

    used_probe_val_dates = set(probe_stats["val"]["dates"])
    val_candidates = [path for path in splits["val"] if date_from_path(path) not in used_probe_val_dates]
    val = build_sample("val_2k", val_candidates, MEDIUM_ROWS["val"], MEDIUM_DAYS["val"])
    val["excluded_probe_val_dates"] = sorted(used_probe_val_dates)

    used_probe_test_dates = set(probe_stats["test"]["dates"])
    final_candidates = [path for path in splits["test"] if date_from_path(path) not in used_probe_test_dates]
    test = build_sample("test_final", final_candidates, MEDIUM_ROWS["test"], MEDIUM_DAYS["test"])
    test["locked_holdout"] = True
    test["excluded_probe_test_dates"] = sorted(used_probe_test_dates)
    return {"train": train, "val": val, "test": test}


def write_manifest(full_stats, probe_stats, medium_stats, files):
    available_variants = ["probe"]
    if full_stats:
        available_variants.append("full")
    if medium_stats:
        available_variants.append("medium")
    manifest = {
        "year": YEAR,
        "daily_root": DAILY_LLM_ROOT,
        "file_prefix": FILE_PREFIX,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "observation_policy": OBSERVATION_POLICY,
        "prediction_horizon_minutes": PREDICTION_HORIZON_MINUTES,
        "num_daily_files": len(files),
        "split_policy": "chronological_days_60_20_20",
        "splits": full_stats,
        "probe_policy": "evenly_spaced_dates_stable_hash_whole_chains",
        "probe_splits": probe_stats,
        "medium_policy": "cross_date_whole_chains_val_selection_locked_final_test",
        "medium_splits": medium_stats,
        "available_variants": available_variants,
    }
    path = os.path.join(OUT_DIR, "llm_dataset_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Build chronological full and cross-date probe splits.")
    parser.add_argument("--probe-only", action="store_true", help="Keep existing full JSONL and only rebuild probe files.")
    parser.add_argument("--build-medium", action="store_true", help="Also build the 10k/2k/5k medium datasets.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    files = list_daily_jsonl()
    splits = split_files(files)
    full_stats = {}
    if not args.probe_only:
        for name in ["train", "val", "test"]:
            path, count, positives = merge_split(name, splits[name])
            full_stats[name] = {
                "rows": count,
                "path": path,
                "num_days": len(splits[name]),
                "date_start": date_from_path(splits[name][0]),
                "date_end": date_from_path(splits[name][-1]),
                "positive_rate": positives / max(1, count),
            }
            print(f"{name}: days={len(splits[name])}, rows={count:,}, positive_rate={positives/max(1,count):.4f}")
    else:
        old_manifest = os.path.join(OUT_DIR, "llm_dataset_manifest.json")
        if os.path.exists(old_manifest):
            with open(old_manifest, encoding="utf-8") as f:
                previous = json.load(f)
            compatible = (
                previous.get("schema_version") == SCHEMA_VERSION
                and previous.get("prompt_version") == PROMPT_VERSION
                and "full" in previous.get("available_variants", [])
            )
            if compatible:
                full_stats = previous.get("splits", {})

    probe_stats = {}
    for name in ["train", "val", "test"]:
        probe_stats[name] = build_probe(name, splits[name])
        stat = probe_stats[name]
        print(f"{name}_probe: days={stat['num_days']}, rows={stat['rows']:,}, positive_rate={stat['positive_rate']:.4f}")

    medium_stats = {}
    if args.build_medium:
        medium_stats = build_medium(splits, probe_stats)
        for name, stat in medium_stats.items():
            print(f"{name}_medium: days={stat['num_days']}, rows={stat['rows']:,}, positive_rate={stat['positive_rate']:.4f}")

    manifest_path = write_manifest(full_stats, probe_stats, medium_stats, files)
    print("Saved manifest:", manifest_path)


if __name__ == "__main__":
    main()
