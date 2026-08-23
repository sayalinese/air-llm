# -*- coding: utf-8 -*-
"""阶段2 真LLM微调分类: 软提示注入 + LoRA, 无门控融合。
GAT 冻结(读 fused_*.pt); 只训 投影器+头+文本层LoRA。
必须用 D:\\vllm\\python\\python.exe 运行。
"""
import math
import os
import torch
from torch import nn

import config2 as C
from llm_text import load_prompts
from soft_model import SoftPromptNet


def _load(split, prompts):
    b = torch.load(os.path.join(C.FUSED_DIR, f"fused_{split}.pt"), weights_only=False)
    z = b['z'].float(); ids = b['sample_id']; y = b['label'].float()
    cap = C.MAX_SAMPLES if split == "train" else (C.VAL_CAP if split == "val" else -1)
    if 0 < cap < len(z):
        z, ids, y = z[:cap], ids[:cap], y[:cap]
    texts = [prompts.get(int(i), "") for i in ids]
    return z, texts, y


def run():
    from transformers import Gemma4ForConditionalGeneration, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from sklearn.metrics import roc_auc_score

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(C.GEMMA_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    full = Gemma4ForConditionalGeneration.from_pretrained(
        C.GEMMA_PATH, torch_dtype=torch.bfloat16).to(dev)
    txt = full.model.language_model                          # 只用文本层
    txt.gradient_checkpointing_enable()
    txt.enable_input_require_grads()
    txt = get_peft_model(txt, LoraConfig(
        r=C.LORA_R, lora_alpha=C.LORA_ALPHA, target_modules=C.TARGET_MODULES,
        lora_dropout=C.LORA_DROPOUT, bias="none"))
    H = txt.config.hidden_size
    net = SoftPromptNet(H).to(dev, torch.bfloat16)
    pad_id = tok.pad_token_id

    prompts = load_prompts(C.TAB_CSV)
    tr = _load("train", prompts); va = _load("val", prompts); te = _load("test", prompts)
    print(f"train={len(tr[0])} val={len(va[0])} test={len(te[0])} H={H}")

    params = [p for p in txt.parameters() if p.requires_grad] + list(net.parameters())
    opt = torch.optim.AdamW(params, lr=C.LR)
    crit = nn.BCEWithLogitsLoss()
    save_every = max(1, math.ceil(len(tr[0]) / C.BATCH / C.VAL_EVERY))  # 每轮评估固定次数

    def batchify(data, i0, i1):
        z, texts, y = data
        enc = tok(texts[i0:i1], padding=True, truncation=True,
                  max_length=C.MAX_LEN, return_tensors="pt")
        return (z[i0:i1].to(dev, torch.bfloat16), enc.input_ids.to(dev),
                enc.attention_mask.to(dev), y[i0:i1].to(dev))

    @torch.no_grad()
    def eval_set(data):
        txt.eval(); net.eval()
        probs, ys = [], []
        for i in range(0, len(data[0]), C.BATCH):
            z, ids, m, y = batchify(data, i, i + C.BATCH)
            logit = net(txt, z, ids, m, pad_id).float()
            probs += torch.sigmoid(logit).cpu().tolist(); ys += y.cpu().tolist()
        return roc_auc_score(ys, probs)

    best = {"auc": -1.0, "wait": 0}
    os.makedirs(C.MODEL2_DIR, exist_ok=True)
    for ep in range(C.EPOCHS):
        txt.train(); net.train()
        n = len(tr[0]); last = 0.0; step = 0
        for i in range(0, n, C.BATCH):
            z, ids, m, y = batchify(tr, i, i + C.BATCH)
            logit = net(txt, z, ids, m, pad_id).float()
            loss = crit(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
            last = loss.item(); step += 1
            if step % save_every == 0:   # 轮内定期验证 + 保存最优
                va_auc = eval_set(va)
                print(f"ep{ep+1} step{step} loss={last:.4f} val_auc={va_auc:.4f}")
                if va_auc > best["auc"]:
                    best = {"auc": va_auc, "wait": 0}
                    torch.save(net.state_dict(), os.path.join(C.MODEL2_DIR, "stage2_fuse.pt"))
                    txt.save_pretrained(os.path.join(C.MODEL2_DIR, "gemma_lora"))
                else:
                    best["wait"] += 1
                    if best["wait"] >= C.PATIENCE:
                        print(f"early stop @ ep{ep+1} step{step} best_auc={best['auc']:.4f}")
                        break
        else:
            continue
        break
    print(f"best_val_auc={best['auc']:.4f}")
    print("test_auc=", round(eval_set(te), 4))


if __name__ == "__main__":
    run()
