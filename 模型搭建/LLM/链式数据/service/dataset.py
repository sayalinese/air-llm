"""链张量的缓存读写与 Dataset/DataLoader 封装。"""
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import cache_path, encoders_path, texts_path, airseq_texts_path


def save_encoders(sparse_maps, vocab_sizes, dense_mean, dense_std):
    os.makedirs(os.path.dirname(encoders_path()), exist_ok=True)
    with open(encoders_path(), "wb") as f:
        pickle.dump(
            {
                "sparse_maps": sparse_maps,
                "vocab_sizes": vocab_sizes,
                "dense_mean": dense_mean,
                "dense_std": dense_std,
            },
            f,
        )


def load_encoders():
    with open(encoders_path(), "rb") as f:
        return pickle.load(f)


def save_split_tensors(split, tensors):
    os.makedirs(os.path.dirname(cache_path(split)), exist_ok=True)
    torch.save(tensors, cache_path(split))


def load_split_tensors(split):
    path = cache_path(split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"链张量缓存缺失: {path}; 请先运行 0_数据整理.py")
    return torch.load(path, weights_only=False)


class ChainDataset(Dataset):
    def __init__(self, tensors):
        self.dense = tensors["dense"]
        self.sparse = tensors["sparse"]
        self.label = tensors["label"]
        self.valid_len = tensors["valid_len"]

    def __len__(self):
        return self.dense.size(0)

    def __getitem__(self, idx):
        return (
            self.dense[idx],
            self.sparse[idx],
            self.label[idx],
            self.valid_len[idx],
        )


def make_loader(tensors, batch_size, shuffle):
    return DataLoader(ChainDataset(tensors), batch_size=batch_size, shuffle=shuffle)


# ---- 链上下文文本 (供 LLM 融合) ----
def save_texts(split, texts):
    os.makedirs(os.path.dirname(texts_path(split)), exist_ok=True)
    with open(texts_path(split), "wb") as f:
        pickle.dump(texts, f)


def load_texts(split):
    path = texts_path(split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"链上下文文本缺失: {path}; 请重新运行 0_数据整理.py")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_airseq_texts(split, texts):
    os.makedirs(os.path.dirname(airseq_texts_path(split)), exist_ok=True)
    with open(airseq_texts_path(split), "wb") as f:
        pickle.dump(texts, f)


def load_airseq_texts(split):
    path = airseq_texts_path(split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"真序列文本缺失: {path}; 请重新运行 0_数据整理.py")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_llm_tensor(sid_out, valid_len, emb_map, hidden_size):
    """按链张量位置对齐 LLM 嵌入 -> float16 张量 [N,L,H], pad/缺失位置置零。"""
    sid_out = np.asarray(sid_out, dtype=object)
    valid_len = np.asarray(valid_len)
    n, L = sid_out.shape
    out = np.zeros((n, L, hidden_size), dtype=np.float16)
    for i in range(n):
        for t in range(int(valid_len[i])):
            vec = emb_map.get(str(sid_out[i, t]))
            if vec is not None:
                out[i, t] = vec.astype(np.float16)
    return torch.from_numpy(out)


class FusionDataset(Dataset):
    def __init__(self, tensors, llm_emb):
        self.dense = tensors["dense"]
        self.sparse = tensors["sparse"]
        self.label = tensors["label"]
        self.valid_len = tensors["valid_len"]
        self.llm_emb = llm_emb  # [N,L,H] float16

    def __len__(self):
        return self.dense.size(0)

    def __getitem__(self, idx):
        return (
            self.dense[idx],
            self.sparse[idx],
            self.label[idx],
            self.valid_len[idx],
            self.llm_emb[idx],
        )


def make_fusion_loader(tensors, llm_emb, batch_size, shuffle):
    return DataLoader(FusionDataset(tensors, llm_emb), batch_size=batch_size, shuffle=shuffle)


def build_airseq_seq_tensors(sid_out, valid_len, store, hidden_size, max_neighbors):
    """对齐真序列嵌入 -> summary[N,L,H], neighbors[N,L,K,H], nb_mask[N,L,K] (均 pad/缺失置零)。"""
    sid_out = np.asarray(sid_out, dtype=object)
    valid_len = np.asarray(valid_len)
    n, L = sid_out.shape
    summary = np.zeros((n, L, hidden_size), dtype=np.float16)
    neighbors = np.zeros((n, L, max_neighbors, hidden_size), dtype=np.float16)
    nb_mask = np.zeros((n, L, max_neighbors), dtype=np.float32)
    for i in range(n):
        for t in range(int(valid_len[i])):
            item = store.get(str(sid_out[i, t]))
            if item is None:
                continue
            s_vec, nb_vec = item
            summary[i, t] = s_vec.astype(np.float16)
            k = min(int(nb_vec.shape[0]), max_neighbors)
            if k > 0:
                neighbors[i, t, :k] = nb_vec[:k].astype(np.float16)
                nb_mask[i, t, :k] = 1.0
    return {
        "summary": torch.from_numpy(summary),
        "neighbors": torch.from_numpy(neighbors),
        "nb_mask": torch.from_numpy(nb_mask),
    }


class FusionSeqDataset(Dataset):
    def __init__(self, tensors, seq):
        self.dense = tensors["dense"]
        self.sparse = tensors["sparse"]
        self.label = tensors["label"]
        self.valid_len = tensors["valid_len"]
        self.summary = seq["summary"]        # [N,L,H]
        self.neighbors = seq["neighbors"]    # [N,L,K,H]
        self.nb_mask = seq["nb_mask"]        # [N,L,K]

    def __len__(self):
        return self.dense.size(0)

    def __getitem__(self, idx):
        return (
            self.dense[idx], self.sparse[idx], self.label[idx], self.valid_len[idx],
            self.summary[idx], self.neighbors[idx], self.nb_mask[idx],
        )


def make_fusion_seq_loader(tensors, seq, batch_size, shuffle):
    return DataLoader(FusionSeqDataset(tensors, seq), batch_size=batch_size, shuffle=shuffle)
