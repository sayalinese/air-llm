import glob
import json
import os


YEAR = 2024
DAILY_LLM_ROOT = r"C:\Users\16960\Desktop\期末论文\三模态数据库建立说明\scripts\Aeolus_V2\dataset\Flight_Chain_LLM_strict"
OUT_DIR = r"C:\Users\16960\Desktop\期末论文\模型搭建\LLM\data"
FILE_PREFIX = "flight_chain_llm_strict_"


def list_daily_jsonl():
    files = sorted(glob.glob(os.path.join(DAILY_LLM_ROOT, str(YEAR), "*", f"{FILE_PREFIX}*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No daily LLM jsonl found under {DAILY_LLM_ROOT}\\{YEAR}")
    return files


def date_from_path(path):
    return os.path.basename(path).replace(FILE_PREFIX, "").replace(".jsonl", "")


def split_dates(files):
    dates = [date_from_path(path) for path in files]
    unique_dates = sorted(set(dates))
    if len(unique_dates) < 5:
        raise ValueError(
            f"Need full enhanced daily files before splitting; found only {len(unique_dates)} day(s) under {DAILY_LLM_ROOT}"
        )
    c1, c2 = int(len(unique_dates) * 0.6), int(len(unique_dates) * 0.8)
    return {
        "train": set(unique_dates[:c1]),
        "val": set(unique_dates[c1:c2]),
        "test": set(unique_dates[c2:]),
    }


def merge_split(split_name, date_set, files):
    out_path = os.path.join(OUT_DIR, f"{split_name}.jsonl")
    count = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for path in files:
            day = date_from_path(path)
            if day not in date_set:
                continue
            with open(path, encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line.rstrip("\n") + "\n")
                        count += 1

    return out_path, count


def write_manifest(stats, files):
    manifest = {
        "year": YEAR,
        "daily_root": DAILY_LLM_ROOT,
        "file_prefix": FILE_PREFIX,
        "schema_version": "chain_llm_strict",
        "prompt_version": "propagation_capsule_strict_v1",
        "observation_policy": "strict_actual_event_before_prediction",
        "num_daily_files": len(files),
        "splits": stats,
    }
    with open(os.path.join(OUT_DIR, "llm_dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = list_daily_jsonl()
    splits = split_dates(files)

    stats = {}
    for name in ["train", "val", "test"]:
        out_path, count = merge_split(name, splits[name], files)
        stats[name] = {
            "rows": count,
            "path": out_path,
            "num_days": len(splits[name]),
        }
        print(f"{name}: days={len(splits[name])}, rows={count:,}, path={out_path}")

    write_manifest(stats, files)
    print("Saved manifest:", os.path.join(OUT_DIR, "llm_dataset_manifest.json"))


if __name__ == "__main__":
    main()
