# -*- coding: utf-8 -*-
"""生成按日期均匀抽样的阶段2显式切片。

输出会原子覆盖 data/阶段2切片/{4k,40k,80k}/{train,val}.pt。
4K: train 1-20日各200条，val 21-25日各800条。
40K: train 1-20日各2000条，val 21-25日各4000条（共20K）。
80K: train 1-20日各4000条，val 21-25日各4000条（共20K）。
"""
import json
import os
import sys

import pandas as pd
import torch

STAGE2_SERVICE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "阶段2")
sys.path.insert(0, STAGE2_SERVICE)

import config2 as C


SPECS = {
    "4k": {"train": 200, "val": 800},
    "40k": {"train": 2000, "val": 4000},
    "80k": {"train": 4000, "val": 4000},
}


def _atomic_torch_save(blob, path):
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def _sample_by_day(blob, id_to_day, per_day, seed):
    ids = [int(x) for x in blob["sample_id"]]
    days = torch.tensor([id_to_day[x] for x in ids], dtype=torch.long)
    labels = blob["label"]
    selected = []

    for day in sorted(days.unique().tolist()):
        candidates = torch.where(days == day)[0]
        if len(candidates) < per_day:
            raise ValueError(f"day={day} only has {len(candidates)} rows, need {per_day}")
        gen = torch.Generator().manual_seed(seed + int(day))
        selected.append(candidates[torch.randperm(len(candidates), generator=gen)[:per_day]])

    idx = torch.cat(selected)
    shuffle_gen = torch.Generator().manual_seed(seed + 1000)
    idx = idx[torch.randperm(len(idx), generator=shuffle_gen)]
    return {
        "z": blob["z"][idx].clone(),
        "sample_id": [ids[i] for i in idx.tolist()],
        "label": labels[idx].clone(),
        "day": days[idx].clone(),
        "source_index": idx.clone(),
    }


def run():
    table = pd.read_csv(C.TAB_CSV, usecols=["航班ID", "日"])
    id_to_day = dict(zip(table["航班ID"].astype(int), table["日"].astype(int)))
    sources = {
        split: torch.load(os.path.join(C.FUSED_DIR, f"fused_{split}.pt"), weights_only=False)
        for split in ("train", "val")
    }

    for tag, split_specs in SPECS.items():
        out_dir = os.path.join(C.SLICE_DIR, tag)
        os.makedirs(out_dir, exist_ok=True)
        manifest = {"seed": C.SEED, "source": C.FUSED_DIR, "splits": {}}

        for split, per_day in split_specs.items():
            sampled = _sample_by_day(sources[split], id_to_day, per_day, C.SEED)
            out_path = os.path.join(out_dir, f"{split}.pt")
            _atomic_torch_save(sampled, out_path)
            days = sorted(sampled["day"].unique().tolist())
            positive_rate = float(sampled["label"].float().mean())
            manifest["splits"][split] = {
                "rows": len(sampled["label"]),
                "per_day": per_day,
                "days": {str(day): int((sampled["day"] == day).sum()) for day in days},
                "positive_rate": positive_rate,
            }
            print(f"[{tag}/{split}] rows={len(sampled['label'])} "
                  f"pos={positive_rate:.4f} -> {out_path}")

        manifest_path = os.path.join(out_dir, "manifest.json")
        tmp_manifest = manifest_path + ".tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        os.replace(tmp_manifest, manifest_path)


if __name__ == "__main__":
    run()
