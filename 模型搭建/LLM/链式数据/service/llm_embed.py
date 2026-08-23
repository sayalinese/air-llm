"""Gemma 文本塔加载 + 链上下文文本嵌入 (冻结, 按 sample_id 一次性缓存)。

复刻外层 service/model.py 的文本塔加载与 last-token 池化, 使链式融合与外层实验口径一致。
冻结文本塔 -> 每条文本的向量恒定 -> 只需跑一次前向并缓存, 之后融合训练/重启直接复用。
"""
import gc
import hashlib
import os

import numpy as np
import torch

from .config import (
    GEMMA_MODEL_PATH,
    LLM_EMBED_CACHE_DIR,
    LLM_MAX_LEN,
    LLM_PROMPT_VERSION,
)


def signature(prompt_version=None, max_len=None):
    pv = prompt_version or LLM_PROMPT_VERSION
    ml = max_len or LLM_MAX_LEN
    raw = f"{os.path.basename(os.path.normpath(GEMMA_MODEL_PATH))}|{pv}|{ml}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def cache_file(split, prompt_version=None, max_len=None):
    return os.path.join(LLM_EMBED_CACHE_DIR, f"text_emb_{split}_{signature(prompt_version, max_len)}.pt")


def _load_store(path, sig):
    if not os.path.exists(path):
        return {}, None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}, None
    if blob.get("signature") != sig:
        return {}, None
    return blob.get("embeddings", {}), blob.get("hidden_size")


def _save_store(path, store, hidden_size, sig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"signature": sig, "hidden_size": hidden_size, "embeddings": store}, path)


def load_tokenizer(path=None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path or GEMMA_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_text_tower(device, path=None):
    """只加载多模态 checkpoint 里的语言模型文本塔 (bfloat16)。"""
    from safetensors import safe_open
    from transformers import Gemma4Config, Gemma4ForCausalLM

    model_path = path or GEMMA_MODEL_PATH
    config = Gemma4Config.from_pretrained(model_path)
    model = Gemma4ForCausalLM(config.text_config).to(dtype=torch.bfloat16)
    weights_path = os.path.join(model_path, "model.safetensors")
    prefix = "model.language_model."
    target_keys = set(model.state_dict().keys())
    state_dict = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                target_key = "model." + key[len(prefix):]
                if target_key in target_keys:
                    state_dict[target_key] = f.get_tensor(key)
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()
    model.to(device)
    model.eval()
    del state_dict
    gc.collect()
    print(f"[llm_embed] loaded Gemma text tower: {model.__class__.__name__}")
    return model


@torch.no_grad()
def _embed_batch(model, tokenizer, texts, device, max_len):
    enc = tokenizer(
        list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True, return_dict=True, use_cache=False,
    )
    hidden = outputs.hidden_states[-1]
    # 取每条序列最后一个非 pad token 的隐状态
    reversed_mask = attention_mask.flip(dims=[1]).long()
    last_from_right = torch.argmax(reversed_mask, dim=1)
    last_indices = attention_mask.size(1) - 1 - last_from_right
    rows = torch.arange(hidden.size(0), device=hidden.device)
    pooled = hidden[rows, last_indices].float().cpu().numpy()
    return pooled


def load_model_and_tokenizer(device=None, path=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return load_text_tower(device, path), load_tokenizer(path)


def missing_ids(split, texts, prompt_version=None, max_len=None):
    store, _ = _load_store(cache_file(split, prompt_version, max_len), signature(prompt_version, max_len))
    return [s for s in texts if s not in store]


def ensure_embeddings(split, texts, device=None, batch_size=None, max_len=None, text_model=None, tokenizer=None, prompt_version=None):
    """确保 texts (dict sid->prompt) 的嵌入齐全并返回。

    传入 text_model/tokenizer 时复用 (跨 split 只加载一次); 否则缺失时本地临时加载并释放。
    prompt_version/max_len 区分不同提示的独立缓存。返回 (emb_map, hidden_size)。
    """
    from .config import LLM_EMBED_BATCH

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or LLM_EMBED_BATCH
    max_len = max_len or LLM_MAX_LEN
    sig = signature(prompt_version, max_len)

    path = cache_file(split, prompt_version, max_len)
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

            iterator = tqdm(range(0, len(missing), batch_size), desc=f"[llm_embed:{split}] embedding")
        except Exception:
            iterator = range(0, len(missing), batch_size)
        for start in iterator:
            chunk = missing[start:start + batch_size]
            pooled = _embed_batch(text_model, tokenizer, [texts[s] for s in chunk], device, max_len)
            for s, vec in zip(chunk, pooled):
                store[s] = vec.astype(np.float16)
            hidden_size = int(pooled.shape[1])
        _save_store(path, store, hidden_size, sig)
        if owns:
            del text_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[llm_embed] {split}: computed {len(missing)} new, total {len(store)} -> {os.path.basename(path)}")
    else:
        print(f"[llm_embed] {split}: all {len(sids)} hit cache, 跳过 Gemma 前向")

    emb_map = {s: np.asarray(store[s], dtype=np.float32) for s in sids}
    if hidden_size is None and emb_map:
        hidden_size = int(next(iter(emb_map.values())).shape[0])
    return emb_map, hidden_size
