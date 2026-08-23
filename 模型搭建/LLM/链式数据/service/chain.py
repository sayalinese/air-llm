"""按官方口径把逐航班记录组装成航班链张量。

对齐 Flight_chain.py:
- 分组键 (OP_CARRIER, OP_CARRIER_FL_NUM, FL_DATE), 组内按计划起飞时间升序;
- 定长 max_len 截断/补零 (右侧补 0, 用 valid_len 记录真实长度);
- 输出 dense[N,L,7] / sparse[N,L,8] / label[N,L] / valid_len[N]。
"""
import numpy as np
import torch

from .config import GROUP_KEYS, SORT_KEY
from .features import encode_dense, encode_sparse


def build_tensors(df, sparse_maps, dense_mean, dense_std, max_len, max_chains, seed, target):
    """从已派生列的 DataFrame 构建链张量。target: 'DEP' | 'ARR'。"""
    df = df.reset_index(drop=True)
    gid = df.groupby(GROUP_KEYS, sort=False).ngroup().to_numpy()
    uniq = np.unique(gid)

    rng = np.random.RandomState(seed)
    if max_chains is not None and len(uniq) > max_chains:
        chosen = rng.choice(uniq, size=max_chains, replace=False)
        keep = np.isin(gid, chosen)
        df = df.loc[keep].reset_index(drop=True)
        gid = df.groupby(GROUP_KEYS, sort=False).ngroup().to_numpy()

    dense = encode_dense(df, dense_mean, dense_std)   # [n,7]
    sparse = encode_sparse(df, sparse_maps)           # [n,8]
    label_col = "label_dep" if target.upper() == "DEP" else "label_arr"
    label = df[label_col].to_numpy(dtype=np.float32)
    sort_values = df[SORT_KEY].to_numpy()
    sample_ids = df["sample_id"].astype(str).to_numpy()  # 对齐钩子: 供后续 LLM 融合按航班取文本/复用缓存

    groups = df.groupby(gid, sort=False).indices  # {gid: positions}
    num_chains = len(groups)
    dense_dim, sparse_dim = dense.shape[1], sparse.shape[1]

    dense_out = np.zeros((num_chains, max_len, dense_dim), dtype=np.float32)
    sparse_out = np.zeros((num_chains, max_len, sparse_dim), dtype=np.int64)
    label_out = np.zeros((num_chains, max_len), dtype=np.float32)
    valid_len = np.zeros((num_chains,), dtype=np.int64)
    sid_out = np.full((num_chains, max_len), "", dtype=object)

    for chain_idx, positions in enumerate(groups.values()):
        ordered = positions[np.argsort(sort_values[positions], kind="mergesort")]
        take = min(len(ordered), max_len)
        sel = ordered[:take]
        dense_out[chain_idx, :take] = dense[sel]
        sparse_out[chain_idx, :take] = sparse[sel]
        label_out[chain_idx, :take] = label[sel]
        sid_out[chain_idx, :take] = sample_ids[sel]
        valid_len[chain_idx] = take

    return {
        "dense": torch.from_numpy(dense_out),
        "sparse": torch.from_numpy(sparse_out),
        "label": torch.from_numpy(label_out),
        "valid_len": torch.from_numpy(valid_len),
        "sample_id": sid_out,
        "num_chains": int(num_chains),
        "num_flights": int(valid_len.sum()),
        "pos_rate": float(_valid_pos_rate(label_out, valid_len)),
    }


def _valid_pos_rate(label_out, valid_len):
    total = int(valid_len.sum())
    if total == 0:
        return 0.0
    pos = 0
    for i, v in enumerate(valid_len):
        pos += int(label_out[i, :v].sum())
    return pos / total


def flatten_valid(tensors):
    """把链张量的有效位置拍平成逐航班样本, 供 XGB 基线使用。

    返回 X[float, n_flights, 7+8], y[n_flights]。稀疏列以整数索引作为数值特征
    (与全项目 XGB 一致: 类别做整数映射后当数值喂树)。
    """
    dense = tensors["dense"].numpy()
    sparse = tensors["sparse"].numpy().astype(np.float32)
    label = tensors["label"].numpy()
    valid_len = tensors["valid_len"].numpy()

    xs, ys = [], []
    for i, v in enumerate(valid_len):
        if v == 0:
            continue
        feat = np.concatenate([dense[i, :v], sparse[i, :v]], axis=1)  # [v, 15]
        xs.append(feat)
        ys.append(label[i, :v])
    if not xs:
        return np.zeros((0, dense.shape[2] + sparse.shape[2]), np.float32), np.zeros((0,), np.float32)
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.float32)
