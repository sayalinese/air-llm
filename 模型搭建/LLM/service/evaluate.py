"""Batched A/B-logit evaluation with validation-only threshold selection."""
import argparse
from collections import defaultdict
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from service.config import (
    ABLATION_STAGE,
    CLASS_NAMES,
    DATASET_VARIANT,
    EVAL_BATCH_SIZE,
    EVAL_MAX_SAMPLES,
    EXPERIMENT_NAME,
    EXPECTED_PREDICTION_HORIZON_MINUTES,
    PROMPT_STYLE,
    SAVE_DIR,
    data_path,
)
from service.dataset import encode_prompt, load_jsonl, validate_label_tokens
from service.model import load_lora, load_model, load_tokenizer


def _safe_auc(y_true, y_prob):
    return None if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_prob)


def _safe_pr_auc(y_true, y_prob):
    return None if len(np.unique(y_true)) < 2 else average_precision_score(y_true, y_prob)


def _batch_prompt_tensors(tokenizer, items, device):
    encoded = [encode_prompt(tokenizer, item, reserve_tokens=1) for item in items]
    width = max(len(ids) for ids in encoded)
    input_ids = []
    attention_mask = []
    for ids in encoded:
        pad = width - len(ids)
        input_ids.append([tokenizer.pad_token_id] * pad + ids)
        attention_mask.append([0] * pad + [1] * len(ids))
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
    )


def get_label_probs(model, tokenizer, jsonl_path, max_samples=None, batch_size=None):
    device = next(model.parameters()).device
    data = load_jsonl(jsonl_path, max_samples=max_samples)
    label_ids = validate_label_tokens(tokenizer)
    class_token_ids = torch.tensor([label_ids[0], label_ids[1]], device=device)
    batch_size = batch_size or EVAL_BATCH_SIZE
    all_probs = []

    for start in tqdm(range(0, len(data), batch_size), desc=f"Scoring {os.path.basename(jsonl_path)}"):
        items = data[start:start + batch_size]
        input_ids, attention_mask = _batch_prompt_tensors(tokenizer, items, device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
            class_logits = logits.float().index_select(-1, class_token_ids)
            all_probs.extend(torch.softmax(class_logits, dim=-1)[:, 1].cpu().tolist())

    labels = np.array([int(item["label"]) for item in data], dtype=np.int8)
    return np.asarray(all_probs), labels, data


def search_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for threshold in np.arange(0.05, 0.951, 0.01):
        score = f1_score(y_true, y_prob >= threshold, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, threshold
    return float(best_t), float(best_f1)


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "auc": _safe_auc(y_true, y_prob),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }, y_pred


def _feature(item, *path, default="缺失"):
    value = item.get("features", {})
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value in (None, "") else str(value)


def build_slices(items):
    definitions = {
        "month": lambda item: _feature(item, "current", "month"),
        "chain_position": lambda item: str(item.get("chain_position", "缺失")),
        "chain_valid_length": lambda item: str(item.get("chain_valid_length", "缺失")),
        "carrier": lambda item: _feature(item, "current", "carrier"),
        "origin": lambda item: _feature(item, "current", "origin"),
        "previous_arrived": lambda item: _feature(item, "tail_chain", "prev_arrived"),
    }
    slices = defaultdict(lambda: defaultdict(list))
    for idx, item in enumerate(items):
        for name, getter in definitions.items():
            slices[name][getter(item)].append(idx)
    return slices


def compute_slice_metrics(items, y_true, y_prob, threshold, min_count=50):
    report = {}
    for slice_name, buckets in build_slices(items).items():
        report[slice_name] = {}
        for bucket, indices in sorted(buckets.items()):
            if len(indices) < min_count:
                continue
            idx = np.asarray(indices)
            metrics, _ = compute_metrics(y_true[idx], y_prob[idx], threshold)
            report[slice_name][bucket] = {
                "count": int(len(indices)),
                "positive_rate": float(y_true[idx].mean()),
                **{key: metrics[key] for key in ["acc", "auc", "pr_auc", "f1", "precision", "recall"]},
            }
    return report


def evaluate(use_base=False, fixed_threshold=None, val_only=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer()
    label_ids = validate_label_tokens(tokenizer)
    print(f"Label tokens: A={label_ids[0]}, B={label_ids[1]}")
    print(
        f"Experiment: {EXPERIMENT_NAME}, stage={ABLATION_STAGE}, "
        f"dataset={DATASET_VARIANT}, batch={EVAL_BATCH_SIZE}"
    )

    if use_base:
        print("Loading base model for zero-shot evaluation...")
        model = load_model(device)
    else:
        print("Loading LoRA model...")
        model = load_lora()
        model.to(device)
    model.eval()

    print("Searching threshold on val set...")
    val_prob, val_true, _ = get_label_probs(model, tokenizer, data_path("val"), EVAL_MAX_SAMPLES)
    searched_t, searched_val_f1 = search_threshold(val_true, val_prob)
    best_t = searched_t if fixed_threshold is None else fixed_threshold
    best_val_f1 = f1_score(val_true, val_prob >= best_t, zero_division=0)
    val_metrics, _ = compute_metrics(val_true, val_prob, best_t)

    if val_only:
        metrics = {
            "model": "base" if use_base else "lora",
            "experiment_name": EXPERIMENT_NAME,
            "dataset_variant": DATASET_VARIANT,
            "ablation_stage": ABLATION_STAGE,
            "prompt_style": PROMPT_STYLE,
            "prediction_horizon_minutes": EXPECTED_PREDICTION_HORIZON_MINUTES,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "threshold_mode": "fixed" if fixed_threshold is not None else "val_best_f1",
            "searched_val_threshold": searched_t,
            "searched_val_f1": searched_val_f1,
            "val": val_metrics,
        }
        os.makedirs(SAVE_DIR, exist_ok=True)
        metrics_path = os.path.join(SAVE_DIR, "metrics_val.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"VAL AUC: {val_metrics['auc']:.4f}")
        print(f"VAL PR-AUC: {val_metrics['pr_auc']:.4f}")
        print(f"VAL F1: {val_metrics['f1']:.4f} at threshold {best_t:.2f}")
        print(f"Validation metrics saved to {metrics_path}")
        return

    print("Evaluating test set with fixed val threshold...")
    test_prob, test_true, test_items = get_label_probs(model, tokenizer, data_path("test"), EVAL_MAX_SAMPLES)
    test_metrics, test_pred = compute_metrics(test_true, test_prob, best_t)
    slices = compute_slice_metrics(test_items, test_true, test_prob, best_t)

    print(f"\nPrompt style: {PROMPT_STYLE}")
    if fixed_threshold is None:
        print(f"Val best threshold: {best_t:.2f} (val_f1={best_val_f1:.4f})")
    else:
        print(
            f"Fixed threshold: {best_t:.2f} (val_f1={best_val_f1:.4f}; "
            f"searched_best={searched_t:.2f}, searched_val_f1={searched_val_f1:.4f})"
        )
    print(f"TEST ACC: {test_metrics['acc']:.4f}")
    print(f"TEST AUC: {test_metrics['auc']:.4f}" if test_metrics["auc"] is not None else "TEST AUC: n/a")
    print(f"TEST PR-AUC: {test_metrics['pr_auc']:.4f}" if test_metrics["pr_auc"] is not None else "TEST PR-AUC: n/a")
    print(f"TEST F1:  {test_metrics['f1']:.4f}")
    print(classification_report(test_true, test_pred, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]], zero_division=0))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.asarray(test_metrics["confusion_matrix"]))

    metrics = {
        "model": "base" if use_base else "lora",
        "experiment_name": EXPERIMENT_NAME,
        "dataset_variant": DATASET_VARIANT,
        "ablation_stage": ABLATION_STAGE,
        "prompt_style": PROMPT_STYLE,
        "prediction_horizon_minutes": EXPECTED_PREDICTION_HORIZON_MINUTES,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "threshold_mode": "fixed" if fixed_threshold is not None else "val_best_f1",
        "searched_val_threshold": searched_t,
        "searched_val_f1": searched_val_f1,
        "val": val_metrics,
        "test": test_metrics,
        "slices": slices,
        **test_metrics,
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    metrics_name = "metrics_base.json" if use_base else "metrics.json"
    metrics_path = os.path.join(SAVE_DIR, metrics_name)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {metrics_path}")

    try:
        from service.plot import plot_metrics
        plot_metrics(metrics_path=metrics_path)
    except Exception as exc:
        print(f"Plot skipped: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gemma flight-delay classifier.")
    parser.add_argument("--base", action="store_true", help="Evaluate base model without LoRA adapter.")
    parser.add_argument("--threshold", type=float, default=None, help="Use a fixed threshold instead of val F1 search.")
    parser.add_argument("--val-only", action="store_true", help="Evaluate val without reading the locked test set.")
    args = parser.parse_args()
    evaluate(use_base=args.base, fixed_threshold=args.threshold, val_only=args.val_only)
