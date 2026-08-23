# -*- coding: utf-8 -*-
"""阶段1 训练: 读 张量预处理数据 -> 训练三模态GAT -> 评估+保存。

每个 split 是不同天的航班图 (inductive); val 选最优, test 评估;
fused 模式额外导出融合向量供阶段2。
"""
import os
import json
import copy

import numpy as np
import torch
from torch import nn

import config
from gat import TriModalGAT

EDGE_NAMES = ("static", "chain", "network")


def _seed(s=config.SEED):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_graph(period, split, device):
    d = os.path.join(config.PROC_DIR, period, split)
    blob = torch.load(os.path.join(d, "nodes.pt"), weights_only=False)
    edges = {}
    for n in EDGE_NAMES:
        ei = np.load(os.path.join(d, f"edge_{n}.npy"))
        edges[n] = torch.from_numpy(ei.astype(np.int64)).to(device)
    return {"dense": blob["dense"].float().to(device),
            "sparse": blob["sparse"].long().to(device),
            "label": blob["label"].float().to(device),
            "sample_id": blob["sample_id"], "edges": edges,
            "n": blob["dense"].size(0)}


def _metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    pr = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else 0.0
    bf, bt = 0.0, 0.5
    for t in np.linspace(0.1, 0.9, 33):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, float(t)
    return {"auc": auc, "pr_auc": pr, "f1": bf, "threshold": bt}


@torch.no_grad()
def _eval(model, g):
    model.eval()
    logit, _ = model(g["dense"], g["sparse"], g["edges"])
    return torch.sigmoid(logit).cpu().numpy(), g["label"].cpu().numpy()


def train_one(mode, tr, va, te, vocab_sizes, device, epochs, export_fused=False):
    _seed()
    model = TriModalGAT(vocab_sizes, dense_dim=config.DENSE_DIM, emb_dim=config.EMB_DIM,
                        hidden=config.HIDDEN, heads=config.HEADS, layers=config.LAYERS,
                        dropout=config.DROPOUT, mode=mode).to(device)
    print(f"\n==== mode={mode} | params={sum(p.numel() for p in model.parameters()):,} ====")

    pos = float(tr["label"].mean().item())
    pw = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=device).clamp(max=10.0)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WD)
    sched = None
    if getattr(config, "USE_COSINE", False):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = {"auc": -1, "state": None, "epoch": -1}
    wait = 0
    for ep in range(epochs):
        model.train()
        logit, _ = model(tr["dense"], tr["sparse"], tr["edges"])
        loss = crit(logit, tr["label"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()

        vp, vt = _eval(model, va)
        vm = _metrics(vt, vp)
        print(f"  ep{ep+1:02d} loss={loss.item():.4f} val_auc={vm['auc']:.4f}")
        if vm["auc"] > best["auc"]:
            best = {"auc": vm["auc"], "state": copy.deepcopy(model.state_dict()), "epoch": ep + 1}
            wait = 0
        else:
            wait += 1
            if wait >= config.PATIENCE:
                print(f"  early stop @ ep{ep+1}")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"stage1_{mode}.pt"))

    tp, tt = _eval(model, te)
    tm = _metrics(tt, tp)
    tm["mode"] = mode
    print(f"  [test] auc={tm['auc']:.4f} pr={tm['pr_auc']:.4f} f1={tm['f1']:.4f}")

    if export_fused and mode == "fused":
        for split, g in (("train", tr), ("val", va), ("test", te)):
            with torch.no_grad():
                _, z = model(g["dense"], g["sparse"], g["edges"])
            torch.save({"z": z.half().cpu(), "sample_id": g["sample_id"], "label": g["label"].cpu()},
                       os.path.join(config.MODEL_DIR, f"fused_{split}.pt"))
    return tm


def run(period, modes, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device} period={period} modes={modes} epochs={epochs}")

    with open(os.path.join(config.PROC_DIR, period, "meta.json"), encoding="utf-8") as f:
        vocab_sizes = json.load(f)["vocab_sizes"]

    tr = load_graph(period, "train", device)
    va = load_graph(period, "val", device)
    te = load_graph(period, "test", device)
    print(f"train={tr['n']:,} val={va['n']:,} test={te['n']:,}")

    rows = []
    for m in modes:
        tm = train_one(m, tr, va, te, vocab_sizes, device, epochs, export_fused=(m == "fused"))
        rows.append(tm)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    import pandas as pd
    df = pd.DataFrame(rows)[["mode", "auc", "pr_auc", "f1", "threshold"]]
    df.to_csv(os.path.join(config.MODEL_DIR, "stage1_compare.csv"), index=False, encoding="utf-8-sig")
    print("\n==== 阶段1 单模态 vs 融合 (test) ====")
    print(df.round(4).to_string(index=False))
    return df
