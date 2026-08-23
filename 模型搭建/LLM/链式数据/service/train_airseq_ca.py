"""真序列跨注意力融合训练: 冻结Gemma 按邻居切片池化 -> LSTM h_t 对邻居序列跨注意力 -> 融合。

自带消融: lstm_only(纯结构) vs crossattn(结构+跨注意力邻居+summary残差), 同数据/同头。
序列嵌入用独立缓存(AIRSEQ_SEQ_PROMPT_VERSION), 一次性编码后训练很快。
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
    AIRSEQ_CA_DIM, AIRSEQ_CA_HEADS, AIRSEQ_EPOCHS, AIRSEQ_MAX_LEN, AIRSEQ_MAX_NEIGHBORS,
    AIRSEQ_SEQ_PROMPT_VERSION, BATCH_SIZE, DENSE_COLS, DROPOUT, EARLY_STOP_PATIENCE, EMB_DIM,
    FUSION_HEAD_HIDDEN, HIDDEN_SIZE, LEARNING_RATE, NUM_LAYERS, POS_WEIGHT, RANDOM_SEED, RNN_TYPE,
    SAVE_DIR, TARGET, WEIGHT_DECAY,
)
from .airseq_embed import ensure_airseq_seq_embeddings, missing_seq_ids
from .dataset import (
    build_airseq_seq_tensors, load_airseq_texts, load_encoders, load_split_tensors,
    make_fusion_seq_loader,
)
from .llm_embed import load_model_and_tokenizer
from .metrics import compute_metrics, print_metrics, save_metrics, search_threshold


def _seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def _mask(valid_len, seq_len, device):
    return torch.arange(seq_len, device=device)[None, :] < valid_len[:, None]


def _to_device(batch, device):
    dense, sparse, label, valid_len, summary, neighbors, nb_mask = batch
    return (dense.to(device), sparse.to(device), label.to(device), valid_len.to(device),
            summary.to(device), neighbors.to(device), nb_mask.to(device))


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        dense, sparse, label, valid_len, summary, neighbors, nb_mask = _to_device(batch, device)
        logits = model(dense, sparse, valid_len, summary, neighbors, nb_mask)
        m = _mask(valid_len, logits.size(1), device)
        probs.append(torch.sigmoid(logits)[m].float().cpu().numpy())
        labels.append(label[m].float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def _train_one(mode, prepared, vocab_sizes, llm_dim, device, do_test):
    _seed()
    from .model import AirseqCrossAttn

    train_t, train_seq = prepared["train"]
    val_t, val_seq = prepared["val"]
    train_loader = make_fusion_seq_loader(train_t, train_seq, BATCH_SIZE, shuffle=True)
    val_loader = make_fusion_seq_loader(val_t, val_seq, BATCH_SIZE, shuffle=False)

    model = AirseqCrossAttn(
        vocab_sizes=vocab_sizes, dense_dim=len(DENSE_COLS), emb_dim=EMB_DIM, hidden=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, dropout=DROPOUT, rnn_type=RNN_TYPE, llm_dim=llm_dim,
        ca_dim=AIRSEQ_CA_DIM, ca_heads=AIRSEQ_CA_HEADS, head_hidden=FUSION_HEAD_HIDDEN, mode=mode,
    ).to(device)
    print(f"\n==== mode={mode} | params={sum(p.numel() for p in model.parameters()):,} ====")

    pos_weight = None if POS_WEIGHT is None else torch.tensor([POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best = {"auc": -1.0, "state": None, "epoch": -1}
    patience = 0
    for epoch in range(AIRSEQ_EPOCHS):
        model.train()
        total, nb = 0.0, 0
        for batch in tqdm(train_loader, desc=f"[{mode}] epoch {epoch + 1}/{AIRSEQ_EPOCHS}"):
            dense, sparse, label, valid_len, summary, neighbors, nb_mask = _to_device(batch, device)
            logits = model(dense, sparse, valid_len, summary, neighbors, nb_mask)
            m = _mask(valid_len, logits.size(1), device)
            loss = criterion(logits[m], label[m])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            nb += 1
        val_prob, val_true = _evaluate(model, val_loader, device)
        vm, _ = compute_metrics(val_true, val_prob, 0.5)
        print(f"[{mode}] epoch {epoch + 1} | loss={total / max(1, nb):.4f} | val_auc={vm['auc']:.4f} pr_auc={vm['pr_auc']:.4f}")
        if vm["auc"] > best["auc"]:
            best = {"auc": vm["auc"], "state": copy.deepcopy(model.state_dict()), "epoch": epoch + 1}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"[{mode}] early stop @ epoch {epoch + 1} (best {best['epoch']}, auc={best['auc']:.4f})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"airseq_ca_{mode}.pt"))

    val_prob, val_true = _evaluate(model, val_loader, device)
    threshold, _ = search_threshold(val_true, val_prob)
    val_full, val_pred = compute_metrics(val_true, val_prob, threshold)
    val_full.update({"model": f"airseq_ca_{mode}", "split": "val", "target": TARGET,
                     "best_epoch": best["epoch"], "num_flights": int(len(val_true))})
    print_metrics(f"AIRSEQ-CA[{mode}] VAL (th={threshold:.2f})", val_full, val_true, val_pred)
    save_metrics(SAVE_DIR, f"metrics_airseq_ca_{mode}_val.csv", val_full)
    rows = [val_full]

    if do_test:
        test_t, test_seq = prepared["test"]
        test_loader = make_fusion_seq_loader(test_t, test_seq, BATCH_SIZE, shuffle=False)
        test_prob, test_true = _evaluate(model, test_loader, device)
        test_full, test_pred = compute_metrics(test_true, test_prob, threshold)
        test_full.update({"model": f"airseq_ca_{mode}", "split": "test", "target": TARGET,
                          "val_threshold": threshold, "num_flights": int(len(test_true))})
        print_metrics(f"AIRSEQ-CA[{mode}] TEST (th={threshold:.2f})", test_full, test_true, test_pred)
        save_metrics(SAVE_DIR, f"metrics_airseq_ca_{mode}_test.csv", test_full)
        rows.append(test_full)
    return rows


def run(do_test=True, modes=None):
    modes = modes or ["lstm_only", "crossattn"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device: {device} | AIRSEQ cross-attn -> {SAVE_DIR} | modes={modes} | K={AIRSEQ_MAX_NEIGHBORS}")

    vocab = load_encoders()["vocab_sizes"]
    splits = ("train", "val", "test") if do_test else ("train", "val")
    texts_by_split = {s: load_airseq_texts(s) for s in splits}
    need = any(missing_seq_ids(s, texts_by_split[s], AIRSEQ_SEQ_PROMPT_VERSION, AIRSEQ_MAX_LEN, AIRSEQ_MAX_NEIGHBORS)
               for s in splits)
    text_model, tokenizer = load_model_and_tokenizer(device) if need else (None, None)

    prepared, llm_dim = {}, None
    for s in splits:
        tensors = load_split_tensors(s)
        store, hidden = ensure_airseq_seq_embeddings(
            s, texts_by_split[s], device=device, max_len=AIRSEQ_MAX_LEN,
            max_neighbors=AIRSEQ_MAX_NEIGHBORS, text_model=text_model, tokenizer=tokenizer,
            prompt_version=AIRSEQ_SEQ_PROMPT_VERSION,
        )
        seq = build_airseq_seq_tensors(tensors["sample_id"], tensors["valid_len"], store, hidden, AIRSEQ_MAX_NEIGHBORS)
        prepared[s] = (tensors, seq)
        llm_dim = hidden
        print(f"[{s}] flights={tensors['num_flights']} pos={tensors['pos_rate']:.4f} llm_dim={hidden}")

    if text_model is not None:
        import gc

        del text_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_rows = []
    for mode in modes:
        all_rows.extend(_train_one(mode, prepared, vocab, llm_dim, device, do_test))

    summary = pd.DataFrame(all_rows)[["model", "split", "auc", "pr_auc", "f1", "acc", "num_flights"]]
    summary.to_csv(os.path.join(SAVE_DIR, "compare_airseq_ca.csv"), index=False, encoding="utf-8-sig")
    print("\n==================== 真序列跨注意力消融对比 (val/test) ====================")
    print(summary.round(4).to_string(index=False))

    with open(os.path.join(SAVE_DIR, "airseq_ca_config.json"), "w", encoding="utf-8") as f:
        json.dump({"modes": modes, "target": TARGET, "llm_dim": llm_dim, "ca_dim": AIRSEQ_CA_DIM,
                   "ca_heads": AIRSEQ_CA_HEADS, "max_neighbors": AIRSEQ_MAX_NEIGHBORS,
                   "prompt_version": AIRSEQ_SEQ_PROMPT_VERSION, "max_len": AIRSEQ_MAX_LEN},
                  f, ensure_ascii=False, indent=2)
    return summary
