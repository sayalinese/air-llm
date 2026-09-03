"""真序列跨注意力用嵌入: 冻结 Gemma 读完整段序列, 按邻居切片池化出每邻居一个向量。

与 llm_embed 的 last-token 池化不同: 这里保留"每个邻居航班一个向量"(≤K 个) + 一个整段 summary,
供融合端用 LSTM 的 h_t 对邻居序列做跨注意力 (不再把序列压成单一向量)。
按 sample_id 一次性缓存 (冻结 -> 向量恒定)。缓存结构: {sid: (summary[H] f16, neighbors[k,H] f16)}。
"""
import gc
import hashlib
import os

import numpy as np
import torch

from .config import GEMMA_MODEL_PATH, LLM_EMBED_BATCH, LLM_EMBED_CACHE_DIR, LLM_MAX_LEN
from .llm_embed import load_model_and_tokenizer


def signature(prompt_version, max_len, k):
    raw = f"{os.path.basename(os.path.normpath(GEMMA_MODEL_PATH))}|{prompt_version}|{max_len}|K{k}|seqpool"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def cache_file(split, prompt_version, max_len, k):
    return os.path.join(LLM_EMBED_CACHE_DIR, f"airseq_seq_{split}_{signature(prompt_version, max_len, k)}.pt")


def _load_store(path, sig):
    if not os.path.exists(path):
        return {}, None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}, None
    if blob.get("signature") != sig:
        return {}, None
    return blob.get("store", {}), blob.get("hidden_size")


def _save_store(path, store, hidden_size, sig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"signature": sig, "hidden_size": hidden_size, "store": store}, path)


def _parse_prompt(prompt):
    """\u5c06 airseq \u63d0\u793a\u62c6\u4e3a (header, [neighbor_line,...])\u3002\u90bb\u5c45\u884c\u4ee5 '  - ' \u5f00\u5934\u3002"""
    header, nbs = [], []
    for line in prompt.split("\n"):
        if line.lstrip().startswith("- "):
            nbs.append(line.strip())
        else:
            header.append(line)
    return "\n".join(header), nbs


@torch.no_grad()
def _encode_batch(model, tokenizer, prompts, device, max_len, k_max):
    bos = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id or 0
    batch_ids, batch_spans = [], []
    for p in prompts:
        header, nbs = _parse_prompt(p)
        ids = [bos] if bos is not None else []
        h_ids = tokenizer(header + "\n", add_special_tokens=False)["input_ids"]
        h_ids = h_ids[:max(0, max_len - len(ids))]
        h_start = len(ids)
        ids = ids + h_ids
        h_end = len(ids)
        nb_spans = []
        for nb in nbs[:k_max]:
            n_ids = tokenizer(nb + "\n", add_special_tokens=False)["input_ids"]
            if len(ids) + len(n_ids) > max_len:
                break
            s = len(ids)
            ids = ids + n_ids
            nb_spans.append((s, len(ids)))
        batch_ids.append(ids)
        batch_spans.append(((h_start, h_end), nb_spans))

    maxT = max(len(x) for x in batch_ids)
    input_ids = torch.full((len(batch_ids), maxT), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch_ids), maxT), dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
    input_ids = input_ids.to(device)
    attn = attn.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attn,
                    output_hidden_states=True, return_dict=True, use_cache=False)
    hidden = outputs.hidden_states[-1]  # [B,T,H]

    results = []
    for i, ((hs, he), nb_spans) in enumerate(batch_spans):
        last = int(attn[i].sum().item()) - 1
        summary = hidden[i, last].float().cpu().numpy().astype(np.float16)
        if nb_spans:
            vecs = [hidden[i, s:e].mean(dim=0) for (s, e) in nb_spans]
            neighbors = torch.stack(vecs, dim=0).float().cpu().numpy().astype(np.float16)
        else:
            neighbors = np.zeros((0, hidden.size(-1)), dtype=np.float16)
        results.append((summary, neighbors))
    return results, int(hidden.size(-1))


def ensure_airseq_seq_embeddings(split, texts, device=None, batch_size=None, max_len=None,
                                 max_neighbors=16, text_model=None, tokenizer=None, prompt_version="airseq_seq_v1"):
    """\u786e\u4fdd texts(dict sid->prompt) \u7684\u5e8f\u5217\u5d4c\u5165\u9f50\u5168\u5e76\u8fd4\u56de (store, hidden_size)\u3002"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or LLM_EMBED_BATCH
    max_len = max_len or LLM_MAX_LEN
    sig = signature(prompt_version, max_len, max_neighbors)
    path = cache_file(split, prompt_version, max_len, max_neighbors)
    store, hidden_size = _load_store(path, sig)
    sids = list(texts.keys())
    missing = [s for s in sids if s not in store]

    if missing:
        owns = False
        if text_model is None:
            text_model, tokenizer = load_model_and_tokenizer(device)
            owns = True
        try:
            from tqdm import tqdm

            iterator = tqdm(range(0, len(missing), batch_size), desc=f"[airseq_seq:{split}] embedding")
        except Exception:
            iterator = range(0, len(missing), batch_size)
        done_since_save = 0
        for start in iterator:
            chunk = missing[start:start + batch_size]
            res, hidden_size = _encode_batch(text_model, tokenizer, [texts[s] for s in chunk],
                                             device, max_len, max_neighbors)
            for s, item in zip(chunk, res):
                store[s] = item
            done_since_save += len(chunk)
            if done_since_save >= 4000:   # 定期落盘: 崩溃最多丢 4000 条, 重启可断点续编
                _save_store(path, store, hidden_size, sig)
                done_since_save = 0
        _save_store(path, store, hidden_size, sig)
        if owns:
            del text_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[airseq_seq] {split}: computed {len(missing)} new, total {len(store)} -> {os.path.basename(path)}")
    else:
        print(f"[airseq_seq] {split}: all {len(sids)} hit cache, \u8df3\u8fc7 Gemma \u524d\u5411")

    if hidden_size is None and store:
        hidden_size = int(next(iter(store.values()))[0].shape[0])
    return store, hidden_size


def missing_seq_ids(split, texts, prompt_version, max_len, max_neighbors):
    store, _ = _load_store(cache_file(split, prompt_version, max_len, max_neighbors),
                           signature(prompt_version, max_len, max_neighbors))
    return [s for s in texts if s not in store]
