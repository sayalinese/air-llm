# -*- coding: utf-8 -*-
"""步骤3: 预处理/编码 -> 落盘模型可读张量 (点边分离)

输入: 新架构/data/{month数据|2024数据}/ 的 表格/链式/图 CSV
输出: 新架构/data/processed/{month|year}/{train,val,test}/
        nodes.pt            {dense,sparse,label,sample_id}
        edge_static.npy     [2,E] 静态图
        edge_chain.npy      [2,E] 链式图
        edge_network.npy    [2,E] 网络(传播)图
      + meta.json           词表大小 + dense 均值/标准差

特征: 8 sparse + 7 dense = 15 (对齐论文)
标签: 出发延误 >= 15 分钟
划分: 按日切 (month: 1-20 train / 21-25 val / 26-30 test)

用法: python preprocess.py month | year
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch

BASE = r"C:\Users\16960\Desktop\期末论文"
DATA = BASE + r"\新架构\data"

SPARSE_COLS = ['承运人', '出发机场', '到达机场', '飞机尾号', '年', '月', '日', '星期']
DENSE_COLS = ['计划出发分钟', '计划到达分钟', '计划飞行时长',
              '出发地温度', '出发地降水', '出发地风速', '到达地温度']

SPLIT_DAYS = {'month': {'train': (1, 21), 'val': (21, 26), 'test': (26, 31)}}


def read_csvs(period):
    d = "4月数据" if period == "month" else "2024数据"
    tag = "4月" if period == "month" else "2024"
    tab = pd.read_csv(os.path.join(DATA, d, f"{tag}表格.csv"))
    ch = pd.read_csv(os.path.join(DATA, d, f"{tag}链式.csv"))
    gr = pd.read_csv(os.path.join(DATA, d, f"{tag}图.csv"))
    return tab, ch, gr


def build_vocab(tab):
    vocab = {}
    for c in SPARSE_COLS:
        vals = tab[c].astype(str).unique().tolist()
        vocab[c] = {v: i + 1 for i, v in enumerate(sorted(vals))}  # 0=padding
    return vocab


def encode_sparse(tab, vocab):
    cols = []
    for c in SPARSE_COLS:
        m = tab[c].astype(str).map(vocab[c]).fillna(0).astype(np.int64)
        cols.append(m.to_numpy())
    return np.stack(cols, axis=1)  # [N,8]


def _hist_rate(tab, col):
    """按 (col,日) 聚合, 只用'前一天及之前'算延误率 (防泄漏)。"""
    g = tab.groupby([col, '日'])['label'].agg(['sum', 'count']).reset_index()
    g = g.sort_values([col, '日'])
    ps = g.groupby(col)['sum'].cumsum() - g['sum']
    pc = g.groupby(col)['count'].cumsum() - g['count']
    g['rate'] = np.where(pc > 0, ps / pc.clip(lower=1), np.nan)
    return g[[col, '日', 'rate']]


def add_hist(tab, col, name, prior):
    g = _hist_rate(tab, col)
    tab = tab.merge(g, on=[col, '日'], how='left')
    tab[name] = tab['rate'].fillna(prior)
    return tab.drop(columns=['rate'])


def main(period):
    out_root = os.path.join(DATA, "张量预处理数据", period)
    tab, ch, gr = read_csvs(period)
    tab['label'] = (tab['出发延误'] >= 15).astype(np.int64)

    vocab = build_vocab(tab)
    vocab_sizes = [len(vocab[c]) + 1 for c in SPARSE_COLS]

    # dense 标准化 (用全期统计, 演示用)
    dense_raw = tab[DENSE_COLS].astype(np.float32).to_numpy()
    dmean = dense_raw.mean(0); dstd = dense_raw.std(0) + 1e-6
    dense_all = (dense_raw - dmean) / dstd
    sparse_all = encode_sparse(tab, vocab)
    id_all = tab['航班ID'].to_numpy()
    day_all = tab['日'].to_numpy()
    label_all = tab['label'].to_numpy()

    static = gr[gr['边类型'] == '静态'][['源航班ID', '目标航班ID']].to_numpy()
    network = gr[gr['边类型'] == '传播'][['源航班ID', '目标航班ID']].to_numpy()
    chainE = ch[['前序航班ID', '后续航班ID']].to_numpy()

    id2pos = {v: i for i, v in enumerate(id_all)}

    for split, (lo, hi) in SPLIT_DAYS['month' if period == 'month' else 'month'].items():
        mask = (day_all >= lo) & (day_all < hi)
        idx = np.where(mask)[0]
        local = {v: i for i, v in enumerate(id_all[idx])}
        n = len(idx)

        nodes = {
            'dense': torch.from_numpy(dense_all[idx].astype(np.float32)),
            'sparse': torch.from_numpy(sparse_all[idx]),
            'label': torch.from_numpy(label_all[idx]),
            'sample_id': [int(i) for i in id_all[idx]],
        }
        sdir = os.path.join(out_root, split)
        os.makedirs(sdir, exist_ok=True)
        torch.save(nodes, os.path.join(sdir, "nodes.pt"))

        ecounts = {}
        for name, E in [("static", static), ("chain", chainE), ("network", network)]:
            rows = [ (local[s], local[d]) for s, d in E if s in local and d in local ]
            ei = np.array(rows, dtype=np.int64).T if rows else np.zeros((2, 0), np.int64)
            np.save(os.path.join(sdir, f"edge_{name}.npy"), ei)
            ecounts[name] = ei.shape[1]

        print(f"[{split}] nodes={n} staticE={ecounts['static']} chainE={ecounts['chain']} netE={ecounts['network']}")

    meta = {'vocab_sizes': vocab_sizes, 'dense_mean': dmean.tolist(),
            'dense_std': dstd.tolist(), 'sparse_cols': SPARSE_COLS, 'dense_cols': DENSE_COLS}
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DONE] -> {out_root}")


# ===== 可调变量 (直接改这里) =====
PERIOD = "month"   # month=4月 / year=全年
# ================================

if __name__ == "__main__":
    main(PERIOD)
