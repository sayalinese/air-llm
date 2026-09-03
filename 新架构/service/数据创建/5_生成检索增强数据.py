# -*- coding: utf-8 -*-
"""生成阶段2检索增强所需的离线top-k缓存。

检索库只使用训练切片。train检索排除自身，val/test仅从train检索，
避免把验证集或测试集标签带入检索库。
"""
import json
import os
import sys

import pandas as pd
import torch


SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE2_DIR = os.path.join(SERVICE_DIR, "阶段2")
RAG_DIR = os.path.join(SERVICE_DIR, "检索增强")
sys.path.insert(0, STAGE2_DIR)
sys.path.insert(0, RAG_DIR)

import config_rag as R
from retriever import hybrid_topk, resolve_device


def _load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    blob = torch.load(path, weights_only=False)
    required = {"z", "sample_id", "label"}
    missing = required.difference(blob)
    if missing:
        raise KeyError(f"{path} missing keys: {sorted(missing)}")
    return blob


def _atomic_torch_save(blob, path):
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def _source_paths():
    return {
        "train": os.path.join(R.STAGE2_SLICE_DIR, R.SOURCE_SLICE_TAG, "train.pt"),
        "val": os.path.join(R.STAGE2_SLICE_DIR, R.VAL_SLICE_TAG, "val.pt"),
        "test": os.path.join(R.FUSED_DIR, "fused_test.pt"),
    }


def _build_metadata(ids, table, codebooks):
    """为每条 fused z 附加只在预测时可见的航班元数据。"""
    by_id = table.set_index("航班ID")
    required = ["承运人", "出发机场", "到达机场", "日", "星期",
                "计划出发分钟", "出发地温度", "出发地风速", "出发地降水"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise KeyError(f"{R.TAB_CSV} missing metadata columns: {missing}")
    result = []
    for sample_id in ids:
        sample_id = int(sample_id)
        if sample_id not in by_id.index:
            raise KeyError(f"sample_id={sample_id} not found in {R.TAB_CSV}")
        row = by_id.loc[sample_id]
        carrier = str(row["承运人"])
        origin = str(row["出发机场"])
        destination = str(row["到达机场"])
        result.append({
            "sample_id": sample_id,
            "day": int(row["日"]),
            "weekday": int(row["星期"]),
            "carrier": carrier,
            "origin": origin,
            "destination": destination,
            "carrier_code": codebooks["carrier"][carrier],
            "origin_code": codebooks["origin"][origin],
            "destination_code": codebooks["destination"][destination],
            "departure_minute": float(row["计划出发分钟"]),
            "temperature": float(row["出发地温度"]),
            "wind": float(row["出发地风速"]),
            "precipitation": float(row["出发地降水"]),
        })
    return result


def run():
    if R.METRIC != "cosine":
        raise ValueError(f"Unsupported metric={R.METRIC}")

    paths = _source_paths()
    datasets = {split: _load(path) for split, path in paths.items()}
    table = pd.read_csv(R.TAB_CSV, usecols=[
        "航班ID", "承运人", "出发机场", "到达机场", "日", "星期",
        "计划出发分钟", "出发地温度", "出发地风速", "出发地降水"])
    codebooks = {
        field: {value: i for i, value in enumerate(sorted(table[column].astype(str).unique()))}
        for field, column in (("carrier", "承运人"), ("origin", "出发机场"),
                              ("destination", "到达机场"))
    }
    metadata = {
        split: _build_metadata(datasets[split]["sample_id"], table, codebooks)
        for split in ("train", "val", "test")
    }
    bank = datasets["train"]
    bank_ids = torch.as_tensor(bank["sample_id"], dtype=torch.long)
    bank_labels = bank["label"].float().cpu()
    os.makedirs(R.RETRIEVAL_DATA_DIR, exist_ok=True)

    manifest = {
        "version": 1,
        "metric": R.METRIC,
        "retrieval_strategy": R.RETRIEVAL_STRATEGY,
        "history_mode": R.HISTORY_MODE,
        "top_k": R.TOP_K,
        "vector_recall_k": R.VECTOR_RECALL_K,
        "struct_candidate_cap": R.STRUCT_CANDIDATE_CAP,
        "source_slice_tag": R.SOURCE_SLICE_TAG,
        "val_slice_tag": R.VAL_SLICE_TAG,
        "device": str(resolve_device(R.RETRIEVAL_DEVICE)),
        "bank_source": paths["train"],
        "splits": {},
    }

    for split in ("train", "val", "test"):
        query = datasets[split]
        query_ids = torch.as_tensor(query["sample_id"], dtype=torch.long)
        indices, vector_scores, rank_scores, levels, valid, candidate_count = hybrid_topk(
            query["z"], bank["z"], metadata[split], metadata["train"], R.TOP_K,
            query_labels=query["label"], bank_labels=bank["label"],
            strategy=R.RETRIEVAL_STRATEGY,
            query_batch_size=R.QUERY_BATCH_SIZE,
            vector_recall_k=R.VECTOR_RECALL_K,
            struct_candidate_cap=R.STRUCT_CANDIDATE_CAP,
            device=R.RETRIEVAL_DEVICE,
            config=R,
            seed=R.SEED,
        )
        safe_indices = indices.clamp_min(0)
        retrieved_ids = bank_ids[safe_indices]
        retrieved_days = torch.tensor([int(m["day"]) for m in metadata["train"]])[safe_indices]
        retrieved_labels = bank["label"].float().cpu()[safe_indices]
        retrieved_ids[~valid] = -1
        retrieved_days[~valid] = -1
        retrieved_labels[~valid] = -1
        if split == "train" and bool((valid & (retrieved_ids == query_ids[:, None])).any().item()):
            raise RuntimeError("Self-retrieval detected in train cache")
        query_days = torch.tensor([int(m["day"]) for m in metadata[split]])
        if bool((valid & (retrieved_days >= query_days[:, None])).any().item()):
            raise RuntimeError("Future or same-day retrieval detected")

        out = {
            "version": 1,
            "split": split,
            "metric": R.METRIC,
            "top_k": R.TOP_K,
            "query_sample_id": query_ids,
            "retrieved_bank_index": indices,
            "retrieved_sample_id": retrieved_ids,
            "retrieved_label": retrieved_labels,
            "retrieved_day": retrieved_days,
            "retrieval_level": levels,
            "retrieval_valid": valid,
            "candidate_count": candidate_count,
            "similarity": vector_scores,
            "rank_score": rank_scores,
            "query_day": query_days,
            "query_source": paths[split],
            "bank_source": paths["train"],
        }
        out_path = R.cache_path(split)
        _atomic_torch_save(out, out_path)
        manifest["splits"][split] = {
            "rows": len(query_ids),
            "source": paths[split],
            "cache": out_path,
            "strict_prior_day": True,
            "valid_cases": int(valid.sum()),
            "no_history_queries": int((valid.sum(1) == 0).sum()),
            "same_route_cases": int((levels == 1).sum()),
        }
        print(f"[{split}] rows={len(query_ids)} top_k={R.TOP_K} -> {out_path}")

    manifest_path = os.path.join(R.RETRIEVAL_DATA_DIR, "manifest.json")
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_manifest, manifest_path)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    run()
