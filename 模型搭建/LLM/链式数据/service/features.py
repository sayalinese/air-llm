"""特征读取与编码 (官方链式口径, 供链构建与 XGB 基线共用)。

只用 15 个静态特征; 实际延误 (DEP_DELAY/ARR_DELAY) 仅用于生成标签, 绝不进特征。
稀疏列在训练集上拟合词表 (0 号保留给 pad/未登录), 连续列在训练集上拟合均值/方差。
"""
import numpy as np
import pandas as pd

from .config import DENSE_COLS, GROUP_KEYS, LABEL_THRESHOLD_MINUTES, SORT_KEY, SPARSE_COLS

# 从 csv 需要读取的原始列 (含分组键与派生所需列, 以及仅用于标签的延误列)
_RAW_COLS = [
    "sample_id",
    "FL_DATE",
    "OP_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN_INDEX",
    "DEST_INDEX",
    "FL_MONTH",
    "FL_WEEK",
    "CRS_DEP_TIME_MIN",
    "CRS_ARR_TIME_MIN",
    "O_TEMP",
    "D_TEMP",
    "O_PRCP",
    "D_PRCP",
    "O_WSPD",
    "D_WSPD",
    "FLIGHTS",
    "DEP_DELAY",
    "ARR_DELAY",
]


def read_split_frame(csv_path, max_rows=None):
    """读取一个分片并派生官方口径所需列。"""
    df = pd.read_csv(csv_path, usecols=lambda c: c in _RAW_COLS, nrows=max_rows, low_memory=False)

    # 派生起降小时 (0-23)
    dep_min = pd.to_numeric(df["CRS_DEP_TIME_MIN"], errors="coerce").fillna(0)
    arr_min = pd.to_numeric(df["CRS_ARR_TIME_MIN"], errors="coerce").fillna(0)
    df["CRS_DEP_TIME_HOUR"] = (dep_min // 60).clip(0, 23).astype("int16")
    df["CRS_ARR_TIME_HOUR"] = (arr_min // 60).clip(0, 23).astype("int16")

    # 连续列转数值
    for col in DENSE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 标签 (仅用于监督, 不进特征)
    df["_dep_delay"] = pd.to_numeric(df["DEP_DELAY"], errors="coerce")
    df["_arr_delay"] = pd.to_numeric(df["ARR_DELAY"], errors="coerce")
    df["label_dep"] = (df["_dep_delay"] > LABEL_THRESHOLD_MINUTES).astype("int8")
    df["label_arr"] = (df["_arr_delay"] > LABEL_THRESHOLD_MINUTES).astype("int8")

    # 排序键
    df[SORT_KEY] = dep_min.astype("int32")
    return df


def _clean(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "<UNK>"
    text = str(value).strip()
    return text if text else "<UNK>"


def fit_sparse_maps(train_df):
    """训练集拟合稀疏词表; idx 从 1 起, 0 号留给 pad/未登录。"""
    maps, vocab_sizes = {}, {}
    for col in SPARSE_COLS:
        values = sorted({_clean(v) for v in train_df[col].tolist() if _clean(v) != "<UNK>"})
        maps[col] = {v: i + 1 for i, v in enumerate(values)}
        vocab_sizes[col] = len(values) + 1  # +1 for index 0
    return maps, vocab_sizes


def encode_sparse(df, maps):
    """稀疏列 -> int 矩阵 [n, 8], 未登录 -> 0。"""
    cols = []
    for col in SPARSE_COLS:
        mapping = maps[col]
        cols.append(df[col].map(lambda v: mapping.get(_clean(v), 0)).to_numpy(dtype=np.int64))
    return np.stack(cols, axis=1)


def fit_dense_stats(train_df):
    arr = train_df[DENSE_COLS].to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return mean, std


def encode_dense(df, mean, std):
    """连续列 -> 标准化矩阵 [n, 7], 缺失填 0 (标准化后的 0=均值)。"""
    arr = df[DENSE_COLS].to_numpy(dtype=np.float32)
    arr = (arr - mean) / std
    return np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)


def group_key_frame(df):
    """返回分组键与排序键, 供链构建。"""
    return df[GROUP_KEYS + [SORT_KEY]].copy()
