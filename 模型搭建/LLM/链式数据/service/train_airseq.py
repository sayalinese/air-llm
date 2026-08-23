"""真序列融合训练: 机场近窗口序列(冻结Gemma编码,缓存) 与 链式LSTM 门控融合。

复用 ChainFusionRNN(结构 h_t ⊕ proj(LLM airseq 嵌入) -> 门控头) + 缓存嵌入基建。
自带消融: lstm_only(纯结构) vs concat(结构+真序列), 同数据/同头, 直接判定有无小胜。
airseq 嵌入用独立缓存(AIRSEQ_PROMPT_VERSION / AIRSEQ_MAX_LEN), 与链上下文嵌入不冲突。
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
    AIRSEQ_EPOCHS, AIRSEQ_MAX_LEN, AIRSEQ_PROMPT_VERSION, BATCH_SIZE, DENSE_COLS, DROPOUT,
    EARLY_STOP_PATIENCE, EMB_DIM, FUSION_HEAD_HIDDEN, FUSION_PROJ_DIM, HIDDEN_SIZE, LEARNING_RATE,
    NUM_LAYERS, POS_WEIGHT, RANDOM_SEED, RNN_TYPE, SAVE_DIR, TARGET, WEIGHT_DECAY,
)
from .dataset import (
    build_llm_tensor, load_airseq_texts, load_encoders, load_split_tensors, make_fusion_loader,
)
from .llm_embed import ensure_embeddings, load_model_and_tokenizer, missing_ids
from .metrics import compute_metrics, print_metrics, save_metrics, search_threshold


def _seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def _mask(valid_len, seq_len, device):
    return torch.arange(seq_len, device=device)[None, :] < valid_len[:, None]


def _to_device(batch, device):
    dense, sparse, label, valid_len, llm_emb = batch
    return (dense.to(device), sparse.to(device), label.to(device), valid_len.to(device), llm_emb.to(device))


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        dense, sparse, label, valid_len, llm_emb = _to_device(batch, device)
        logits = model(dense, sparse, valid_len, llm_emb)
        m = _mask(valid_len, logits.size(1), device)
        probs.append(torch.sigmoid(logits)[m].float().cpu().numpy())
        labels.append(label[m].float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def _train_one(mode, prepared, vocab_sizes, llm_dim, device, do_test):
    _seed()
    from .model import ChainFusionRNN

    train_t, train_llm, _ = prepared["train"]
    val_t, val_llm, _ = prepared["val"]
    train_loader = make_fusion_loader(train_t, train_llm, BATCH_SIZE, shuffle=True)
    val_loader = make_fusion_loader(val_t, val_llm, BATCH_SIZE, shuffle=False)

    model = ChainFusionRNN(
        vocab_sizes=vocab_sizes, dense_dim=len(DENSE_COLS), emb_dim=EMB_DIM, hidden=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, dropout=DROPOUT, rnn_type=RNN_TYPE, llm_dim=llm_dim,
        proj_dim=FUSION_PROJ_DIM, head_hidden=FUSION_HEAD_HIDDEN, mode=mode,
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
            dense, sparse, label, valid_len, llm_emb = _to_device(batch, device)
            logits = model(dense, sparse, valid_len, llm_emb)
            m = _mask(valid_len, logits.size(1), device)
            loss = criterion(logits[m], label[m])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            nb += 1
        val_prob, val_true = _evaluate(model, val_loader, device)
        m, _ = compute_metrics(val_true, val_prob, 0.5)
        print(f"[{mode}] epoch {epoch + 1} | loss={total / max(1, nb):.4f} | val_auc={m['auc']:.4f} pr_auc={m['pr_auc']:.4f}")
        if m["auc"] > best["auc"]:
            best = {"auc": m["auc"], "state": copy.deepcopy(model.state_dict()), "epoch": epoch + 1}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"[{mode}] early stop @ epoch {epoch + 1} (best {best['epoch']}, auc={best['auc']:.4f})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"airseq_{mode}.pt"))

    val_prob, val_true = _evaluate(model, val_loader, device)
    threshold, _ = search_threshold(val_true, val_prob)
    val_full, val_pred = compute_metrics(val_true, val_prob, threshold)
    val_full.update({"model": f"airseq_{mode}", "split": "val", "target": TARGET,
                     "best_epoch": best["epoch"], "num_flights": int(len(val_true))})
    print_metrics(f"AIRSEQ[{mode}] VAL (th={threshold:.2f})", val_full, val_true, val_pred)
    save_metrics(SAVE_DIR, f"metrics_airseq_{mode}_val.csv", val_full)
    rows = [val_full]

    if do_test:
        test_t, test_llm, _ = prepared["test"]
        test_loader = make_fusion_loader(test_t, test_llm, BATCH_SIZE, shuffle=False)
        test_prob, test_true = _evaluate(model, test_loader, device)
        test_full, test_pred = compute_metrics(test_true, test_prob, threshold)
        test_full.update({"model": f"airseq_{mode}", "split": "test", "target": TARGET,
                          "val_threshold": threshold, "num_flights": int(len(test_true))})
        print_metrics(f"AIRSEQ[{mode}] TEST (th={threshold:.2f})", test_full, test_true, test_pred)
        save_metrics(SAVE_DIR, f"metrics_airseq_{mode}_test.csv", test_full)
        rows.append(test_full)
    return rows


def run(do_test=True, modes=None):
    modes = modes or ["lstm_only", "concat"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device: {device} | AIRSEQ fusion -> {SAVE_DIR} | modes={modes}")

    vocab = load_encoders()["vocab_sizes"]
    splits = ("train", "val", "test") if do_test else ("train", "val")
    texts_by_split = {s: load_airseq_texts(s) for s in splits}
    need = any(missing_ids(s, texts_by_split[s], AIRSEQ_PROMPT_VERSION, AIRSEQ_MAX_LEN) for s in splits)
    text_model, tokenizer = load_model_and_tokenizer(device) if need else (None, None)

    prepared, llm_dim = {}, None
    for s in splits:
        tensors = load_split_tensors(s)
        emb_map, hidden = ensure_embeddings(
            s, texts_by_split[s], device=device, text_model=text_model, tokenizer=tokenizer,
            prompt_version=AIRSEQ_PROMPT_VERSION, max_len=AIRSEQ_MAX_LEN,
        )
        llm_emb = build_llm_tensor(tensors["sample_id"], tensors["valid_len"], emb_map, hidden)
        prepared[s] = (tensors, llm_emb, hidden)
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
    summary.to_csv(os.path.join(SAVE_DIR, "compare_airseq.csv"), index=False, encoding="utf-8-sig")
    print("\n==================== 真序列消融对比 (val/test) ====================")
    print(summary.round(4).to_string(index=False))

    with open(os.path.join(SAVE_DIR, "airseq_config.json"), "w", encoding="utf-8") as f:
        json.dump({"modes": modes, "target": TARGET, "llm_dim": llm_dim,
                   "prompt_version": AIRSEQ_PROMPT_VERSION, "max_len": AIRSEQ_MAX_LEN}, f, ensure_ascii=False, indent=2)
    return summary
