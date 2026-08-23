"""链式 TCN 基线训练 + 评估 (与 LSTM 同口径: 逐位置 masked BCE, 无未来泄露)。

复用 train_lstm 的 seed/mask/evaluate 等辅助, 仅把模型换成 ChainTCN。
产物独立: chain_tcn.pt / metrics_tcn_val.csv / metrics_tcn_test.csv / tcn_config.json,
便于与 chain_lstm(链式 LSTM)、airseq-crossattn(真序列融合) 同台对比。
"""
import copy
import json
import os

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .config import (
    BATCH_SIZE, DENSE_COLS, DROPOUT, EARLY_STOP_PATIENCE, EMB_DIM, EPOCHS, EXPERIMENT_NAME,
    HIDDEN_SIZE, LEARNING_RATE, POS_WEIGHT, RANDOM_SEED, SAVE_DIR, TARGET, TCN_KERNEL, TCN_LAYERS,
    WEIGHT_DECAY,
)
from .dataset import load_encoders, load_split_tensors, make_loader
from .metrics import compute_metrics, print_metrics, save_metrics, search_threshold
from .train_lstm import _mask, _quick_metrics, _seed, _to_device, evaluate


def train(do_test=True):
    _seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Experiment: {EXPERIMENT_NAME} -> {SAVE_DIR} | TCN")

    enc = load_encoders()
    vocab_sizes = enc["vocab_sizes"]
    train_t = load_split_tensors("train")
    val_t = load_split_tensors("val")
    test_t = load_split_tensors("test") if do_test else None
    print(
        f"Train chains={train_t['num_chains']} flights={train_t['num_flights']} pos={train_t['pos_rate']:.4f} | "
        f"Val chains={val_t['num_chains']} flights={val_t['num_flights']} pos={val_t['pos_rate']:.4f}"
    )

    from .model import ChainTCN  # 延迟导入

    model = ChainTCN(
        vocab_sizes=vocab_sizes,
        dense_dim=len(DENSE_COLS),
        emb_dim=EMB_DIM,
        hidden=HIDDEN_SIZE,
        num_layers=TCN_LAYERS,
        dropout=DROPOUT,
        kernel=TCN_KERNEL,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TCN hidden={HIDDEN_SIZE} layers={TCN_LAYERS} kernel={TCN_KERNEL} params={n_params:,}")

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
        total_loss, n_batches = 0.0, 0
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
        _, vm = _quick_metrics(val_true, val_prob)
        train_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch + 1} | train_loss={train_loss:.4f} | val_auc={vm['auc']:.4f} val_pr_auc={vm['pr_auc']:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **vm})

        if vm["auc"] > best["auc"]:
            best = {"auc": vm["auc"], "state": copy.deepcopy(model.state_dict()), "epoch": epoch + 1}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch + 1} (best epoch {best['epoch']}, val_auc={best['auc']:.4f})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "chain_tcn.pt"))
    pd.DataFrame(history).to_csv(os.path.join(SAVE_DIR, "history_tcn.csv"), index=False, encoding="utf-8-sig")

    val_prob, val_true = evaluate(model, val_loader, device)
    threshold, val_f1 = search_threshold(val_true, val_prob)
    val_full, val_pred = compute_metrics(val_true, val_prob, threshold)
    val_full.update({"model": "chain_tcn", "target": TARGET, "searched_val_f1": val_f1,
                     "best_epoch": best["epoch"], "num_flights": int(len(val_true))})
    print_metrics(f"CHAIN-TCN VAL (threshold={threshold:.2f})", val_full, val_true, val_pred)
    save_metrics(SAVE_DIR, "metrics_tcn_val.csv", val_full)

    result = {"val": val_full, "threshold": threshold}
    if do_test:
        test_loader = make_loader(test_t, BATCH_SIZE, shuffle=False)
        test_prob, test_true = evaluate(model, test_loader, device)
        test_full, test_pred = compute_metrics(test_true, test_prob, threshold)
        test_full.update({"model": "chain_tcn", "target": TARGET, "val_threshold": threshold,
                          "num_flights": int(len(test_true))})
        print_metrics(f"CHAIN-TCN TEST (val threshold={threshold:.2f})", test_full, test_true, test_pred)
        save_metrics(SAVE_DIR, "metrics_tcn_test.csv", test_full)
        result["test"] = test_full

    meta = {"experiment_name": EXPERIMENT_NAME, "target": TARGET, "arch": "TCN",
            "hidden_size": HIDDEN_SIZE, "layers": TCN_LAYERS, "kernel": TCN_KERNEL,
            "best_epoch": best["epoch"], "best_val_auc": best["auc"], "val_threshold": threshold}
    with open(os.path.join(SAVE_DIR, "tcn_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return result
