"""指标与阈值工具 (与外层 xgb_eval / evaluate 对齐, 供 B 各脚本共用)。"""
import os

import numpy as np
import pandas as pd
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

from .config import CLASS_NAMES


def safe_auc(y_true, y_prob):
    return None if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true, y_prob):
    return None if len(np.unique(y_true)) < 2 else float(average_precision_score(y_true, y_prob))


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "auc": safe_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }, y_pred


def search_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for threshold in np.arange(0.05, 0.951, 0.01):
        score = f1_score(y_true, np.asarray(y_prob) >= threshold, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, threshold
    return float(best_t), float(best_f1)


def metric_value(y_true, y_prob, name):
    if name == "auc":
        value = safe_auc(y_true, y_prob)
    elif name == "pr_auc":
        value = safe_pr_auc(y_true, y_prob)
    elif name == "f1":
        _, f1 = search_threshold(y_true, y_prob)
        value = f1
    else:
        raise ValueError(f"Unknown tune metric: {name}")
    return float(value) if value is not None else float("nan")


def print_metrics(title, metrics, y_true, y_pred):
    print(f"\n{title}")
    print(f"ACC:    {metrics['acc']:.4f}")
    print(f"AUC:    {metrics['auc']:.4f}" if metrics["auc"] is not None else "AUC:    NA")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}" if metrics["pr_auc"] is not None else "PR-AUC: NA")
    print(f"F1:     {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
            zero_division=0,
        )
    )
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.asarray(metrics["confusion_matrix"]))


def save_metrics(save_dir, filename, metrics):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    pd.DataFrame([metrics]).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Metrics saved to {path}")
    return path
