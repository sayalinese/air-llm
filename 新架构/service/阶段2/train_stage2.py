# -*- coding: utf-8 -*-
"""阶段2 真LLM微调分类: 软提示注入 + LoRA, 无门控融合。
GAT 冻结(读 fused_*.pt); 只训 投影器+头+文本层LoRA。
必须用 D:\\vllm\\python\\python.exe 运行。
"""
import copy
import csv
import json
import math
import os
import torch
from torch import nn

import config2 as C
from llm_text import load_prompts
from soft_model import SoftPromptNet


def _load(split, prompts):
    slice_tags = {4000: "4k", 40000: "40k", 80000: "80k"}
    if split == "train":
        tag = slice_tags.get(C.MAX_SAMPLES)
    elif split == "val":
        tag = C.VAL_SLICE_TAG
    else:
        tag = None
    slice_path = os.path.join(C.SLICE_DIR, tag, f"{split}.pt") if tag else None
    if C.USE_PREBUILT_SLICES and slice_path:
        if not os.path.exists(slice_path):
            raise FileNotFoundError(
                f"Missing stage2 slice: {slice_path}. Run service/数据创建/4_生成数据划分切片.py first.")
        source = slice_path
    else:
        source = os.path.join(C.FUSED_DIR, f"fused_{split}.pt")
    b = torch.load(source, weights_only=False)
    z = b['z'].float(); ids = b['sample_id']; y = b['label'].float()
    using_slice = slice_path is not None and source == slice_path
    cap = -1 if using_slice else {"train": C.MAX_SAMPLES, "val": C.VAL_CAP, "test": C.TEST_CAP}[split]
    if 0 < cap < len(z):
        z, ids, y = z[:cap], ids[:cap], y[:cap]
    texts = [prompts.get(int(i), "") for i in ids]
    return z, texts, y, source, ids


def run():
    from transformers import Gemma4ForConditionalGeneration, AutoTokenizer
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(C.SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(C.SEED)
    tok = AutoTokenizer.from_pretrained(C.GEMMA_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if C.READOUT_MODE in {"text_then_soft", "cls"}:
        tok.padding_side = "left"
    full = Gemma4ForConditionalGeneration.from_pretrained(
        C.GEMMA_PATH, torch_dtype=torch.bfloat16).to(dev)
    txt = full.model.language_model                          # 只用文本层
    txt.config.use_cache = False
    txt.gradient_checkpointing_enable()
    txt.enable_input_require_grads()
    txt = get_peft_model(txt, LoraConfig(
        r=C.LORA_R, lora_alpha=C.LORA_ALPHA, target_modules=C.TARGET_MODULES,
        lora_dropout=C.LORA_DROPOUT, bias="none"))
    H = txt.config.hidden_size
    net = SoftPromptNet(H).to(dev, torch.bfloat16)
    if C.TOKEN_GATE:
        net.gate_logits.data = net.gate_logits.data.float()  # 门控保fp32: bf16下2e-4更新会被舍入吞掉
    pad_id = tok.pad_token_id

    prompts = load_prompts(C.TAB_CSV)
    tr = _load("train", prompts); va = _load("val", prompts); te = _load("test", prompts)
    print(f"train={len(tr[0])} val={len(va[0])} test={len(te[0])} H={H}")
    print(f"train_source={tr[3]}\nval_source={va[3]}\ntest_source={te[3]}")

    params = [p for p in txt.parameters() if p.requires_grad] + list(net.parameters())
    opt = torch.optim.AdamW(params, lr=C.LR)
    if C.USE_POS_WEIGHT:   # 消融 A: 正样本加权 (同阶段 1 公式)
        pos = float(tr[2].mean().item())
        pw = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=dev).clamp(max=10.0)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        crit = nn.BCEWithLogitsLoss()
    save_every = C.VAL_STEPS if C.VAL_STEPS > 0 else max(
        1, math.ceil(len(tr[0]) / C.BATCH / C.VAL_EVERY))
    sched = None
    if C.USE_SCHED:   # N3: warmup100步 + cosine衰减
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        total = math.ceil(len(tr[0]) / C.BATCH) * C.EPOCHS
        sched = SequentialLR(opt, [LinearLR(opt, start_factor=0.1, total_iters=100),
                                   CosineAnnealingLR(opt, T_max=max(total - 100, 1))],
                             milestones=[100])

    def batchify(data, order, i0, i1):
        z, texts, y = data[:3]
        idx = order[i0:i1]
        enc = tok([texts[k] for k in idx], padding=True, truncation=True,
                  max_length=C.MAX_LEN, return_tensors="pt")
        t = torch.tensor(idx)
        return (z[t].to(dev, torch.bfloat16), enc.input_ids.to(dev),
                enc.attention_mask.to(dev), y[t].to(dev))

    @torch.no_grad()
    def eval_set(data, return_outputs=False):
        txt_was_training, net_was_training = txt.training, net.training
        txt.eval(); net.eval()
        probs, ys = [], []
        order = list(range(len(data[0])))
        for i in range(0, len(data[0]), C.BATCH):
            z, ids, m, y = batchify(data, order, i, i + C.BATCH)
            logit = net(txt, z, ids, m, pad_id).float()
            probs += torch.sigmoid(logit).cpu().tolist(); ys += y.cpu().tolist()
        if txt_was_training:
            txt.train()
        if net_was_training:
            net.train()
        auc = float(roc_auc_score(ys, probs))
        return (auc, probs, ys) if return_outputs else auc

    best = {"auc": -1.0, "wait": 0, "net": None, "lora": None,
            "epoch": 0, "global_step": 0}
    history = []
    os.makedirs(C.MODEL2_DIR, exist_ok=True)
    n = len(tr[0])
    steps_per_epoch = math.ceil(n / C.BATCH)
    min_stop_step = steps_per_epoch if C.MAX_SAMPLES < 0 and C.FULL_PASS_BEFORE_EARLY_STOP else 0
    global_step = 0
    interval_loss = 0.0
    interval_batches = 0
    stop = False

    for ep in range(C.EPOCHS):
        txt.train(); net.train()
        order = torch.randperm(n).tolist() if C.SHUFFLE else list(range(n))
        for step, i in enumerate(range(0, n, C.BATCH), 1):
            z, ids, m, y = batchify(tr, order, i, i + C.BATCH)
            logit = net(txt, z, ids, m, pad_id).float()
            loss = crit(logit, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if sched is not None:
                sched.step()
            global_step += 1
            interval_loss += loss.item()
            interval_batches += 1

            epoch_end = i + C.BATCH >= n
            if global_step % save_every == 0 or epoch_end:
                train_loss = interval_loss / max(interval_batches, 1)
                interval_loss = 0.0
                interval_batches = 0
                va_auc = eval_set(va)
                lr = opt.param_groups[0]["lr"]
                history.append({"epoch": ep + 1, "step": step, "global_step": global_step,
                                "train_loss": train_loss, "val_auc": va_auc, "lr": lr})
                print(f"ep{ep+1} step{step} global{global_step} train_loss={train_loss:.4f} "
                      f"val_auc={va_auc:.4f} lr={lr:.2e}")
                if va_auc > best["auc"]:
                    best = {"auc": va_auc, "wait": 0,
                            "net": copy.deepcopy(net.state_dict()),
                            "lora": copy.deepcopy(get_peft_model_state_dict(txt)),
                            "epoch": ep + 1, "global_step": global_step}
                    torch.save(net.state_dict(), os.path.join(C.MODEL2_DIR, f"stage2_fuse_{C.RUN_TAG}.pt"))
                    txt.save_pretrained(os.path.join(C.MODEL2_DIR, f"gemma_lora_{C.RUN_TAG}"),
                                        save_embedding_layers=False)
                elif global_step >= min_stop_step:
                    best["wait"] += 1
                    if best["wait"] >= C.PATIENCE:
                        print(f"early stop @ ep{ep+1} step{step} global{global_step} "
                              f"best_auc={best['auc']:.4f}")
                        stop = True
                        break
        if stop:
            break

    if best["net"] is not None:
        net.load_state_dict(best["net"])
        set_peft_model_state_dict(txt, best["lora"])
    print(f"best_val_auc={best['auc']:.4f} @ ep{best['epoch']} global{best['global_step']}")
    te_auc_raw, te_probs, te_ys = eval_set(te, return_outputs=True)
    te_auc = round(te_auc_raw, 4)
    print("test_auc=", te_auc)

    history_path = os.path.join(C.MODEL2_DIR, f"history_{C.RUN_TAG}.csv")
    with open(history_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "global_step", "train_loss", "val_auc", "lr"])
        writer.writeheader(); writer.writerows(history)

    if C.SAVE_PREDICTIONS:
        pred_path = os.path.join(C.MODEL2_DIR, f"test_predictions_{C.RUN_TAG}.csv")
        with open(pred_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "label", "probability"])
            writer.writerows(zip(te[4], te_ys, te_probs))

    result = {
        "run_tag": C.RUN_TAG, "max_samples": C.MAX_SAMPLES,
        "readout_mode": C.READOUT_MODE, "train_rows": len(tr[0]),
        "val_rows": len(va[0]), "test_rows": len(te[0]),
        "best_val_auc": best["auc"], "best_epoch": best["epoch"],
        "best_global_step": best["global_step"], "test_auc": te_auc,
        "train_source": tr[3], "val_source": va[3], "test_source": te[3],
    }
    with open(os.path.join(C.MODEL2_DIR, f"result_{C.RUN_TAG}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    run()
