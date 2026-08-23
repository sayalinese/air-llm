"""2_基准模型 (链式对照): 纯 XGBoost, 同 15 静态特征, 逐航班拍平, 无序列结构。

从 0_数据整理 缓存的链张量里取有效位置拍平成逐航班样本 —— 与 3_模型训练 的 LSTM
使用完全相同的航班集合。二者之差 = "序列结构" 带来的增益。这是"链式 LSTM 能否打过
表格 XGB"的公平对照下限。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import xgboost as xgb

from service.config import (
    RANDOM_SEED,
    SAVE_DIR,
    TARGET,
    XGB_COLSAMPLE_BYTREE,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_N_JOBS,
    XGB_SUBSAMPLE,
)
from service.chain import flatten_valid
from service.dataset import load_split_tensors
from service.metrics import compute_metrics, print_metrics, save_metrics, search_threshold


def _params(y):
    pos = max(1, int(y.sum()))
    neg = max(1, int((1 - y).sum()))
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": XGB_MAX_DEPTH,
        "eta": XGB_LEARNING_RATE,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "seed": RANDOM_SEED,
        "scale_pos_weight": neg / pos,
        "nthread": XGB_N_JOBS,
    }


def main():
    print("==================== 2_基准模型: 纯 XGBoost (15 静态特征) ====================")
    x_train, y_train = flatten_valid(load_split_tensors("train"))
    x_val, y_val = flatten_valid(load_split_tensors("val"))
    print(f"Train flights={len(y_train):,} pos={y_train.mean():.4f} | Val flights={len(y_val):,}")

    booster = xgb.train(
        _params(y_train),
        xgb.DMatrix(x_train, label=y_train),
        num_boost_round=XGB_N_ESTIMATORS,
    )

    val_prob = booster.predict(xgb.DMatrix(x_val))
    threshold, val_f1 = search_threshold(y_val, val_prob)
    val_metrics, val_pred = compute_metrics(y_val, val_prob, threshold)
    val_metrics.update({"model": "xgb_flat", "target": TARGET, "searched_val_f1": val_f1, "num_flights": len(y_val)})
    print_metrics(f"XGB VAL (threshold={threshold:.2f})", val_metrics, y_val, val_pred)
    save_metrics(SAVE_DIR, "metrics_xgb_val.csv", val_metrics)

    x_test, y_test = flatten_valid(load_split_tensors("test"))
    test_prob = booster.predict(xgb.DMatrix(x_test))
    test_metrics, test_pred = compute_metrics(y_test, test_prob, threshold)
    test_metrics.update({"model": "xgb_flat", "target": TARGET, "val_threshold": threshold, "num_flights": len(y_test)})
    print_metrics(f"XGB TEST (val threshold={threshold:.2f})", test_metrics, y_test, test_pred)
    save_metrics(SAVE_DIR, "metrics_xgb_test.csv", test_metrics)


if __name__ == "__main__":
    main()
