# -*- coding: utf-8 -*-
"""RAG-Gemma-LoRA训练循环。

检索缓存离线生成；训练阶段只读取缓存和本地prompt，不访问数据库或重新检索。
"""
import copy
import csv
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn

import config_rag as R
from rag_model import RetrievalAugmentedSoftPromptNet
from rag_text import load_retrieval_prompts


def _source_path(split):
    if split == "train":
        return os.path.join(R.STAGE2_SLICE_DIR, R.SOURCE_SLICE_TAG, "train.pt")
    if split == "val":
        return os.path.join(R.STAGE2_SLICE_DIR, R.VAL_SLICE_TAG, "val.pt")
    return os.path.join(R.FUSED_DIR, "fused_test.pt")


def _load_split(split):
    cache_path = R.cache_path(split)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Missing retrieval cache: {cache_path}. Run "
            r"D:\vllm\python\python.exe service\数据创建\5_生成检索增强数据.py first.")
    source = _source_path(split)
    blob = torch.load(source, weights_only=False)
    cache = torch.load(cache_path, weights_only=False)
    cache_ids = [int(x) for x in cache["query_sample_id"]]
    ids = [int(x) for x in blob["sample_id"]]
    if cache_ids != ids:
        raise ValueError(f"RAG cache/query order mismatch for split={split}")
    prompts = load_retrieval_prompts(R.TAB_CSV, cache_path, R.INCLUDE_SIMILARITY_SCORE)
    texts = [prompts.get(sample_id, "") for sample_id in ids]
    if any(not text for text in texts):
        missing = [sample_id for sample_id, text in zip(ids, texts) if not text][:5]
        raise KeyError(f"Missing RAG prompts for sample IDs: {missing}")
    return {
        "z": blob["z"].float(),
        "texts": texts,
        "y": blob["label"].float(),
        "ids": ids,
        "source": source,
    }


def _metric_dict(y_true, probs, threshold=None):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    result = {
        "auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
    }
    if threshold is not None:
        pred = (probs >= threshold).astype(np.int64)
        result.update({
            "acc": float(accuracy_score(y_true, pred)),
            "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "threshold": float(threshold),
        })
    return result


def _select_val_threshold(y_true, probs):
    from sklearn.metrics import f1_score
    best = {"threshold": 0.5, "f1": -1.0}
    for threshold in np.linspace(0.01, 0.99, 197):
        score = float(f1_score(y_true, np.asarray(probs) >= threshold, zero_division=0))
        if score > best["f1"] or (score == best["f1"] and abs(threshold - 0.5) < abs(best["threshold"] - 0.5)):
            best = {"threshold": float(threshold), "f1": score}
    return best


def _retrieval_diagnostics(split):
    cache = torch.load(R.cache_path(split), weights_only=False)
    valid = torch.as_tensor(cache["retrieval_valid"], dtype=torch.bool)
    levels = torch.as_tensor(cache["retrieval_level"], dtype=torch.long)
    labels = torch.as_tensor(cache["retrieved_label"], dtype=torch.float32)
    similarity = torch.as_tensor(cache["similarity"], dtype=torch.float32)
    return {
        "valid_rate": float(valid.float().mean()),
        "no_history_rate": float((valid.sum(1) == 0).float().mean()),
        "same_route_rate": float(((levels == 1) & valid).float().sum() / max(valid.sum().item(), 1)),
        "retrieved_positive_rate": float(labels[valid].mean()) if valid.any() else 0.0,
        "mean_vector_similarity": float(similarity[valid].mean()) if valid.any() else 0.0,
        "mean_candidate_count": float(torch.as_tensor(cache["candidate_count"], dtype=torch.float32).mean()),
    }


def run():
    from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                      set_peft_model_state_dict)
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    torch.manual_seed(R.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(R.SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(R.GEMMA_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "left"

    full = Gemma4ForConditionalGeneration.from_pretrained(
        R.GEMMA_PATH, torch_dtype=torch.bfloat16).to(dev)
    txt = full.model.language_model
    txt.config.use_cache = False
    txt.gradient_checkpointing_enable()
    txt.enable_input_require_grads()
    txt = get_peft_model(txt, LoraConfig(
        r=R.LORA_R, lora_alpha=R.LORA_ALPHA,
        target_modules=R.TARGET_MODULES, lora_dropout=R.LORA_DROPOUT,
        bias="none"))
    H = txt.config.hidden_size
    net = RetrievalAugmentedSoftPromptNet(H).to(dev, torch.bfloat16)
    if R.TOKEN_GATE:
        net.gate_logits.data = net.gate_logits.data.float()
    pad_id = tok.pad_token_id

    tr, va, te = (_load_split(split) for split in ("train", "val", "test"))
    print(f"train={len(tr['z'])} val={len(va['z'])} test={len(te['z'])} H={H}")
    print(f"train_source={tr['source']}\nval_source={va['source']}\ntest_source={te['source']}")
    print(f"retrieval_strategy={R.RETRIEVAL_STRATEGY} history_mode={R.HISTORY_MODE} top_k={R.TOP_K}")
    for split in ("train", "val", "test"):
        print(f"retrieval_{split}={json.dumps(_retrieval_diagnostics(split), ensure_ascii=False)}")

    params = [p for p in txt.parameters() if p.requires_grad] + list(net.parameters())
    opt = torch.optim.AdamW(params, lr=R.LR)
    if R.USE_POS_WEIGHT:
        pos = float(tr["y"].mean().item())
        weight = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=dev).clamp(max=10.0)
        crit = nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        crit = nn.BCEWithLogitsLoss()

    steps_per_epoch = math.ceil(len(tr["z"]) / R.BATCH)
    total_steps = steps_per_epoch * R.EPOCHS
    sched = None
    if R.USE_SCHED:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = min(100, max(total_steps - 1, 1))
        sched = SequentialLR(
            opt,
            [LinearLR(opt, start_factor=0.1, total_iters=warmup),
             CosineAnnealingLR(opt, T_max=max(total_steps - warmup, 1))],
            milestones=[warmup])

    def batchify(data, order, start, end):
        idx = order[start:end]
        enc = tok([data["texts"][i] for i in idx], padding=True,
                  truncation=True, max_length=R.MAX_LEN, return_tensors="pt")
        t = torch.tensor(idx, dtype=torch.long)
        return (data["z"][t].to(dev, torch.bfloat16),
                enc.input_ids.to(dev), enc.attention_mask.to(dev),
                data["y"][t].to(dev))

    @torch.no_grad()
    def evaluate(data, return_outputs=False):
        txt_was_training, net_was_training = txt.training, net.training
        txt.eval(); net.eval()
        probs, ys = [], []
        order = list(range(len(data["z"])))
        for start in range(0, len(order), R.BATCH):
            z, ids, mask, y = batchify(data, order, start, start + R.BATCH)
            logit = net(txt, z, ids, mask, pad_id).float()
            probs.extend(torch.sigmoid(logit).cpu().tolist())
            ys.extend(y.cpu().tolist())
        if txt_was_training:
            txt.train()
        if net_was_training:
            net.train()
        auc = _metric_dict(ys, probs)["auc"]
        return (auc, probs, ys) if return_outputs else auc

    best = {"auc": -1.0, "wait": 0, "net": None, "lora": None,
            "epoch": 0, "global_step": 0}
    history = []
    os.makedirs(R.RETRIEVAL_MODEL_DIR, exist_ok=True)
    n = len(tr["z"])
    min_stop_step = steps_per_epoch if R.FULL_PASS_BEFORE_EARLY_STOP else 0
    global_step = 0
    interval_loss, interval_batches = 0.0, 0
    stop = False
    train_start = time.perf_counter()

    for epoch in range(R.EPOCHS):
        txt.train(); net.train()
        order = torch.randperm(n).tolist() if R.SHUFFLE else list(range(n))
        for step, start in enumerate(range(0, n, R.BATCH), 1):
            z, ids, mask, y = batchify(tr, order, start, start + R.BATCH)
            logit = net(txt, z, ids, mask, pad_id)
            loss = crit(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            global_step += 1
            interval_loss += float(loss.item())
            interval_batches += 1
            epoch_end = start + R.BATCH >= n
            if global_step % R.VAL_STEPS == 0 or epoch_end:
                train_loss = interval_loss / max(interval_batches, 1)
                interval_loss, interval_batches = 0.0, 0
                val_auc = evaluate(va)
                lr = opt.param_groups[0]["lr"]
                history.append({"epoch": epoch + 1, "step": step,
                                "global_step": global_step, "train_loss": train_loss,
                                "val_auc": val_auc, "lr": lr})
                print(f"ep{epoch + 1} step{step} global{global_step} "
                      f"train_loss={train_loss:.4f} val_auc={val_auc:.4f} lr={lr:.2e}")
                if val_auc > best["auc"]:
                    best = {"auc": val_auc, "wait": 0,
                            "net": copy.deepcopy(net.state_dict()),
                            "lora": copy.deepcopy(get_peft_model_state_dict(txt)),
                            "epoch": epoch + 1, "global_step": global_step}
                    torch.save(net.state_dict(), os.path.join(
                        R.RETRIEVAL_MODEL_DIR, f"stage2_rag_fuse_{R.RUN_TAG}.pt"))
                    txt.save_pretrained(os.path.join(
                        R.RETRIEVAL_MODEL_DIR, f"gemma_lora_{R.RUN_TAG}"),
                        save_embedding_layers=False)
                elif global_step >= min_stop_step:
                    best["wait"] += 1
                    if best["wait"] >= R.PATIENCE:
                        print(f"early stop @ ep{epoch + 1} step{step} "
                              f"global{global_step} best_auc={best['auc']:.4f}")
                        stop = True
                        break
        if stop:
            break

    if best["net"] is None:
        raise RuntimeError("No validation checkpoint was saved")
    net.load_state_dict(best["net"])
    set_peft_model_state_dict(txt, best["lora"])
    val_auc, val_probs, val_ys = evaluate(va, return_outputs=True)
    test_auc, test_probs, test_ys = evaluate(te, return_outputs=True)
    threshold = _select_val_threshold(val_ys, val_probs)["threshold"]
    val_metrics = _metric_dict(val_ys, val_probs, threshold)
    test_metrics = _metric_dict(test_ys, test_probs, threshold)
    elapsed = time.perf_counter() - train_start
    print(f"best_val_auc={best['auc']:.4f} @ ep{best['epoch']} global{best['global_step']}")
    print(f"val_auc={val_metrics['auc']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f} "
          f"val_f1={val_metrics['f1']:.4f} threshold={threshold:.3f}")
    print(f"test_auc={test_metrics['auc']:.4f} test_pr_auc={test_metrics['pr_auc']:.4f} "
          f"test_f1={test_metrics['f1']:.4f} threshold={threshold:.3f}")

    history_path = os.path.join(R.RETRIEVAL_MODEL_DIR, f"history_{R.RUN_TAG}.csv")
    with open(history_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "global_step",
                                               "train_loss", "val_auc", "lr"])
        writer.writeheader(); writer.writerows(history)

    if R.SAVE_PREDICTIONS:
        for split, data, probs, ys in (("val", va, val_probs, val_ys),
                                       ("test", te, test_probs, test_ys)):
            path = os.path.join(R.RETRIEVAL_MODEL_DIR,
                                f"{split}_predictions_{R.RUN_TAG}.csv")
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id", "label", "probability", "threshold_prediction"])
                writer.writerows((sid, label, prob, int(prob >= threshold))
                                 for sid, label, prob in zip(data["ids"], ys, probs))

    result = {
        "run_tag": R.RUN_TAG,
        "retrieval_strategy": R.RETRIEVAL_STRATEGY,
        "history_mode": R.HISTORY_MODE,
        "top_k": R.TOP_K,
        "max_len": R.MAX_LEN,
        "train_rows": len(tr["z"]), "val_rows": len(va["z"]), "test_rows": len(te["z"]),
        "best_val_auc": best["auc"], "best_epoch": best["epoch"],
        "best_global_step": best["global_step"],
        "val": val_metrics, "test": test_metrics,
        "train_seconds": elapsed,
        "train_source": tr["source"], "val_source": va["source"],
        "test_source": te["source"],
        "retrieval_cache_dir": R.RETRIEVAL_DATA_DIR,
        "retrieval_diagnostics": {
            split: _retrieval_diagnostics(split)
            for split in ("train", "val", "test")
        },
    }
    result_path = os.path.join(R.RETRIEVAL_MODEL_DIR, f"result_{R.RUN_TAG}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    run()
