"""评估: 提取答案 token 的 logprob → AUC/F1/阈值搜索"""
import argparse
import os
import json
import sys
from collections import defaultdict

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
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
from service.config import DATA_DIR, SAVE_DIR, PROMPT_TEMPLATE, LABEL_MAP, EVAL_MAX_SAMPLES, PROMPT_STYLE
from service.dataset import load_jsonl, build_text_payload
from service.model import load_tokenizer, load_lora, load_model


def _safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_prob)


def _safe_pr_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    return average_precision_score(y_true, y_prob)


def _label_logprob(model, input_ids, label_ids):
    ids = input_ids
    total_logprob = 0.0

    for token_id in label_ids:
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1, :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)

        total_logprob += log_probs[token_id].item()
        next_token = torch.tensor([[token_id]], dtype=torch.long, device=ids.device)
        ids = torch.cat([ids, next_token], dim=1)

    return total_logprob / max(1, len(label_ids))


def get_label_probs(model, tokenizer, jsonl_path, max_samples=None):
    device = next(model.parameters()).device
    data = load_jsonl(jsonl_path, max_samples=max_samples)

    label_ids = {
        label: tokenizer.encode(text, add_special_tokens=False)
        for label, text in LABEL_MAP.items()
    }

    all_probs = []
    all_labels = []

    for item in tqdm(data, desc=f"Scoring {os.path.basename(jsonl_path)}"):
        prompt = PROMPT_TEMPLATE.format(text=build_text_payload(item))
        prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        neg_score = _label_logprob(model, ids, label_ids[0])
        pos_score = _label_logprob(model, ids, label_ids[1])
        probs = torch.softmax(torch.tensor([neg_score, pos_score]), dim=0)
        all_probs.append(probs[1].item())
        all_labels.append(int(item['label']))

    return np.array(all_probs), np.array(all_labels), data


def search_threshold(y_true, y_prob):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        pred = (y_prob > t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob > threshold).astype(int)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'auc': _safe_auc(y_true, y_prob),
        'pr_auc': _safe_pr_auc(y_true, y_prob),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'threshold': float(threshold),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }, y_pred


def print_label_tokens(tokenizer):
    print("Label token check:")
    for label, text in LABEL_MAP.items():
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"  {label}={text}: {token_ids}")


def _extract_field(item, field):
    text = item.get("input") or item.get("text") or ""
    marker = f"{field}："
    start = text.find(marker)
    if start < 0:
        return "缺失"
    start += len(marker)
    end = text.find("\n", start)
    if end < 0:
        end = len(text)
    return text[start:end].strip() or "缺失"


def _bucket_month(item):
    return _extract_field(item, "月份")


def _bucket_has_prev(item):
    text = item.get("input") or item.get("text") or ""
    return "无前序" if "前序航班1：\n无前序航班" in text else "有前序"


def build_slices(items):
    slice_defs = {
        "month": _bucket_month,
        "chain_position": lambda item: str(item.get("chain_position", "缺失")),
        "chain_valid_length": lambda item: str(item.get("chain_valid_length", "缺失")),
        "carrier": lambda item: _extract_field(item, "承运人"),
        "origin": lambda item: _extract_field(item, "出发机场"),
        "has_previous_flight": _bucket_has_prev,
    }
    slices = defaultdict(lambda: defaultdict(list))
    for idx, item in enumerate(items):
        for slice_name, getter in slice_defs.items():
            slices[slice_name][getter(item)].append(idx)
    return slices


def compute_slice_metrics(items, y_true, y_prob, threshold, min_count=100):
    report = {}
    slices = build_slices(items)
    for slice_name, buckets in slices.items():
        report[slice_name] = {}
        for bucket, indices in sorted(buckets.items()):
            if len(indices) < min_count:
                continue
            idx = np.array(indices)
            metrics, _ = compute_metrics(y_true[idx], y_prob[idx], threshold)
            report[slice_name][bucket] = {
                'count': int(len(indices)),
                'positive_rate': float(y_true[idx].mean()),
                'acc': metrics['acc'],
                'auc': metrics['auc'],
                'pr_auc': metrics['pr_auc'],
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
            }
    return report


def evaluate(use_base=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = load_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print_label_tokens(tokenizer)

    if use_base:
        print("Loading base model for zero-shot evaluation...")
        model = load_model(device)
    else:
        print("Loading LoRA model...")
        model = load_lora()
        model.to(device)
    model.to(device)
    model.eval()

    val_path = os.path.join(DATA_DIR, "val.jsonl")
    test_path = os.path.join(DATA_DIR, "test.jsonl")

    print("Searching threshold on val set...")
    val_prob, val_true, _ = get_label_probs(model, tokenizer, val_path, max_samples=EVAL_MAX_SAMPLES)
    best_t, best_val_f1 = search_threshold(val_true, val_prob)
    val_metrics, _ = compute_metrics(val_true, val_prob, best_t)

    print("Evaluating test set with fixed val threshold...")
    test_prob, test_true, test_items = get_label_probs(model, tokenizer, test_path, max_samples=EVAL_MAX_SAMPLES)
    test_metrics, test_pred = compute_metrics(test_true, test_prob, best_t)
    slices = compute_slice_metrics(test_items, test_true, test_prob, best_t)

    print(f"\nPrompt style: {PROMPT_STYLE}")
    print(f"Val best threshold: {best_t:.2f} (val_f1={best_val_f1:.4f})")
    print(f"TEST ACC: {test_metrics['acc']:.4f}")
    print(f"TEST AUC: {test_metrics['auc']:.4f}" if test_metrics['auc'] is not None else "TEST AUC: n/a")
    print(f"TEST PR-AUC: {test_metrics['pr_auc']:.4f}" if test_metrics['pr_auc'] is not None else "TEST PR-AUC: n/a")
    print(f"TEST F1:  {test_metrics['f1']:.4f}")
    print(classification_report(test_true, test_pred, target_names=['正常', '延误'], zero_division=0))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(test_metrics['confusion_matrix']))

    metrics = {
        'model': 'base' if use_base else 'lora',
        'prompt_style': PROMPT_STYLE,
        'eval_max_samples': EVAL_MAX_SAMPLES,
        'val': val_metrics,
        'test': test_metrics,
        'slices': slices,
        **test_metrics,
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    metrics_name = 'metrics_base.json' if use_base else 'metrics.json'
    with open(os.path.join(SAVE_DIR, metrics_name), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {os.path.join(SAVE_DIR, metrics_name)}")

    # 自动画精度图
    try:
        from .plot import plot_metrics
        plot_metrics()
        print("Metrics chart generated.")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Gemma flight-delay classifier.")
    parser.add_argument("--base", action="store_true", help="Evaluate base model without LoRA adapter.")
    args = parser.parse_args()
    evaluate(use_base=args.base)
