"""链式 LSTM 训练 + 评估管线。

逐位置 masked BCE 训练; 每轮在验证集按 AUC 选最优; 最优模型在 val 搜阈值后评测 test。
评测只统计每条链的有效位置 (补零位置屏蔽), 得到逐航班指标, 与 2_基准模型 的 XGB 同口径可比。
"""
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .config import (
    BATCH_SIZE,
    DENSE_COLS,
    DROPOUT,
    EARLY_STOP_PATIENCE,
    EMB_DIM,
    EPOCHS,
    EXPERIMENT_NAME,
    HIDDEN_SIZE,
    LEARNING_RATE,
    NUM_LAYERS,
    POS_WEIGHT,
    RANDOM_SEED,
    RNN_TYPE,
    SAVE_DIR,
    TARGET,
    WEIGHT_DECAY,
)
from .dataset import load_encoders, load_split_tensors, make_loader
from .metrics import compute_metrics, print_metrics, save_metrics, search_threshold


def _seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def _mask(valid_len, seq_len, device):
    ar = torch.arange(seq_len, device=device).unsqueeze(0)
    return ar < valid_len.unsqueeze(1)  # [B,L] bool


def _to_device(batch, device):
    dense, sparse, label, valid_len = batch
    return dense.to(device), sparse.to(device), label.to(device), valid_len.to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        dense, sparse, label, valid_len = _to_device(batch, device)
        logits = model(dense, sparse, valid_len)
        mask = _mask(valid_len, logits.size(1), device)
        p = torch.sigmoid(logits)[mask]
        y = label[mask]
        probs.append(p.float().cpu().numpy())
        labels.append(y.float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train(do_test=True):
    _seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Experiment: {EXPERIMENT_NAME} -> {SAVE_DIR}")

    enc = load_encoders()
    vocab_sizes = enc["vocab_sizes"]
    train_t = load_split_tensors("train")
    val_t = load_split_tensors("val")
    test_t = load_split_tensors("test") if do_test else None
    print(
        f"Train chains={train_t['num_chains']} flights={train_t['num_flights']} pos={train_t['pos_rate']:.4f} | "
        f"Val chains={val_t['num_chains']} flights={val_t['num_flights']} pos={val_t['pos_rate']:.4f}"
    )

    from .model import ChainRNN  # 延迟导入, 避免无 torch 环境下 import 失败

    model = ChainRNN(
        vocab_sizes=vocab_sizes,
        dense_dim=len(DENSE_COLS),
        emb_dim=EMB_DIM,
        hidden=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        rnn_type=RNN_TYPE,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {RNN_TYPE} hidden={HIDDEN_SIZE} layers={NUM_LAYERS} params={n_params:,}")

    pos_weight = None if POS_WEIGHT is None else torch.tensor([POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loader = make_loader(train_t, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_t, BATCH_SIZE, shuffle=False)

    best = {"auc": -1.0, "state": None, "epoch": -1}
    history = []
    patience = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            dense, sparse, label, valid_len = _to_device(batch, device)
            logits = model(dense, sparse, valid_len)
            mask = _mask(valid_len, logits.size(1), device)
            loss = criterion(logits[mask], label[mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        val_prob, val_true = evaluate(model, val_loader, device)
        _, val_metrics = _quick_metrics(val_true, val_prob)
        train_loss = total_loss / max(1, n_batches)
        print(
            f"Epoch {epoch + 1} | train_loss={train_loss:.4f} "
            f"| val_auc={val_metrics['auc']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f}"
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

        if val_metrics["auc"] > best["auc"]:
            best = {"auc": val_metrics["auc"], "state": copy.deepcopy(model.state_dict()), "epoch": epoch + 1}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch + 1} (best epoch {best['epoch']}, val_auc={best['auc']:.4f})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "chain_lstm.pt"))
    pd.DataFrame(history).to_csv(os.path.join(SAVE_DIR, "history.csv"), index=False, encoding="utf-8-sig")

    # 最优模型: val 搜阈值 -> 应用到 test
    val_prob, val_true = evaluate(model, val_loader, device)
    threshold, val_f1 = search_threshold(val_true, val_prob)
    val_full, val_pred = compute_metrics(val_true, val_prob, threshold)
    val_full.update({"model": f"chain_{RNN_TYPE.lower()}", "target": TARGET, "searched_val_f1": val_f1,
                     "best_epoch": best["epoch"], "num_flights": int(len(val_true))})
    print_metrics(f"CHAIN-{RNN_TYPE} VAL (threshold={threshold:.2f})", val_full, val_true, val_pred)
    save_metrics(SAVE_DIR, "metrics_lstm_val.csv", val_full)

    result = {"val": val_full, "threshold": threshold}
    if do_test:
        test_loader = make_loader(test_t, BATCH_SIZE, shuffle=False)
        test_prob, test_true = evaluate(model, test_loader, device)
        test_full, test_pred = compute_metrics(test_true, test_prob, threshold)
        test_full.update({"model": f"chain_{RNN_TYPE.lower()}", "target": TARGET, "val_threshold": threshold,
                          "num_flights": int(len(test_true))})
        print_metrics(f"CHAIN-{RNN_TYPE} TEST (val threshold={threshold:.2f})", test_full, test_true, test_pred)
        save_metrics(SAVE_DIR, "metrics_lstm_test.csv", test_full)
        result["test"] = test_full

    _save_config(best, threshold)
    return result


def _quick_metrics(y_true, y_prob):
    """训练日志用的快速指标 (阈值 0.5 的 acc/f1 + 阈值无关的 auc/pr_auc)。"""
    m, _ = compute_metrics(y_true, y_prob, 0.5)
    return None, {"auc": m["auc"], "pr_auc": m["pr_auc"], "acc": m["acc"], "f1": m["f1"]}


def _save_config(best, threshold):
    meta = {
        "experiment_name": EXPERIMENT_NAME,
        "target": TARGET,
        "rnn_type": RNN_TYPE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "emb_dim": EMB_DIM,
        "epochs": EPOCHS,
        "best_epoch": best["epoch"],
        "best_val_auc": best["auc"],
        "val_threshold": threshold,
        "pos_weight": POS_WEIGHT,
    }
    with open(os.path.join(SAVE_DIR, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
