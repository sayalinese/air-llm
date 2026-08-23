"""6_真序列融合_多种子: 跨注意力融合 5 种子误差棒 (crossattn vs lstm_only)。

复用 train_airseq_ca 的数据准备; 仅改变模型初始化种子, 用于评估稳定性。
用法:
    python 6_真序列融合_多种子.py  # 默认 10k, 5 种子
    python 6_真序列融合_多种子.py --exp chain_airseq_5k  # 5k 规模
    或设置环境变量: CHAIN_EXPERIMENT_NAME=chain_airseq_10k / CHAIN_SEEDS=42,123,456
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from service.config import SAVE_DIR, RANDOM_SEED as _cfg_seed
from service.train_airseq_ca import _train_one, _seed, load_encoders, load_airseq_texts, load_split_tensors
from service.train_airseq_ca import ensure_airseq_seq_embeddings, build_airseq_seq_tensors, load_model_and_tokenizer
from service.train_airseq_ca import missing_seq_ids, AIRSEQ_SEQ_PROMPT_VERSION, AIRSEQ_MAX_LEN, AIRSEQ_MAX_NEIGHBORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=None, help="覆盖 CHAIN_EXPERIMENT_NAME")
    parser.add_argument("--seeds", default=None, help="逗号分隔种子(默认 42,123,456,789,1024)")
    parser.add_argument("--modes", default=None, help="逗号分隔模式(默认 lstm_only,gate,catgate,crossattn)")
    args = parser.parse_args()

    if args.exp:
        os.environ["CHAIN_EXPERIMENT_NAME"] = args.exp

    seeds = [int(s.strip()) for s in (args.seeds or "42,123,456,789,1024").split(",") if s.strip()]
    modes = [m.strip() for m in (args.modes or "lstm_only,gate,catgate,crossattn").split(",") if m.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 重新导入(因为 SAVE_DIR 依赖 CHAIN_EXPERIMENT_NAME)
    from service.config import SAVE_DIR
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"SAVE_DIR={SAVE_DIR} | seeds={seeds} | modes={modes} | K={AIRSEQ_MAX_NEIGHBORS}")

    # ---------- 数据准备(只做一次) ----------
    vocab = load_encoders()["vocab_sizes"]
    splits = ("train", "val", "test")
    texts_by_split = {s: load_airseq_texts(s) for s in splits}
    need = any(missing_seq_ids(s, texts_by_split[s], AIRSEQ_SEQ_PROMPT_VERSION,
                                AIRSEQ_MAX_LEN, AIRSEQ_MAX_NEIGHBORS) for s in splits)
    text_model, tokenizer = load_model_and_tokenizer(device) if need else (None, None)

    prepared, llm_dim = {}, None
    for s in splits:
        tensors = load_split_tensors(s)
        store, hidden = ensure_airseq_seq_embeddings(
            s, texts_by_split[s], device=device, max_len=AIRSEQ_MAX_LEN,
            max_neighbors=AIRSEQ_MAX_NEIGHBORS, text_model=text_model, tokenizer=tokenizer,
            prompt_version=AIRSEQ_SEQ_PROMPT_VERSION,
        )
        seq = build_airseq_seq_tensors(tensors["sample_id"], tensors["valid_len"], store, hidden, AIRSEQ_MAX_NEIGHBORS)
        prepared[s] = (tensors, seq)
        llm_dim = hidden
        print(f"[{s}] flights={tensors['num_flights']} pos={tensors['pos_rate']:.4f} llm_dim={hidden}")

    if text_model is not None:
        import gc
        del text_model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- 多种子训练 ----------
    import service.train_airseq_ca as _tca
    _orig_seed = _tca._seed  # 备份原始 _seed
    all_results = []
    for seed in seeds:
        # 替换 _seed 为当前种子的版本 (解决 Python from-import 整数不可变传递问题)
        def _make_seeder(s):
            def _s():
                torch.manual_seed(s)
                np.random.seed(s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(s)
            return _s
        _tca._seed = _make_seeder(seed)
        print(f"\n{'='*60}\n  SEED={seed}\n{'='*60}")
        for mode in modes:
            rows = _train_one(mode, prepared, vocab, llm_dim, device, do_test=True)
            for r in rows:
                r["seed"] = seed
            all_results.extend(rows)
    _tca._seed = _orig_seed  # 恢复

    # ---------- 汇总 ----------
    df = pd.DataFrame(all_results)[["model", "seed", "split", "auc", "pr_auc", "f1", "acc", "num_flights"]]
    df.to_csv(os.path.join(SAVE_DIR, "multi_seed_results.csv"), index=False, encoding="utf-8-sig")

    # 按模式×分片聚合 mean±std
    agg = df[df["split"] == "test"].groupby("model").agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        pr_mean=("pr_auc", "mean"), pr_std=("pr_auc", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        n=("auc", "count"),
    ).reset_index()
    agg.to_csv(os.path.join(SAVE_DIR, "multi_seed_aggregate.csv"), index=False, encoding="utf-8-sig")

    print("\n==================== 多种子汇总 (test) ====================")
    for _, row in agg.iterrows():
        print(f"  {row['model']:<12} AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f}  "
              f"PR={row['pr_mean']:.4f}±{row['pr_std']:.4f}  "
              f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}  n={int(row['n'])}")

    # 每个融合模式 vs lstm_only 的配对 t 检验 (df=n-1, 双尾)
    _t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}
    base = df[(df["split"] == "test") & (df["model"] == "airseq_ca_lstm_only")].sort_values("seed")
    print("\n----------- 各融合模式 vs lstm_only 配对 t 检验 (test AUC) -----------")
    for mode in modes:
        if mode == "lstm_only":
            continue
        cur = df[(df["split"] == "test") & (df["model"] == f"airseq_ca_{mode}")].sort_values("seed")
        if len(cur) != len(base) or len(base) < 3:
            continue
        diffs = cur["auc"].values - base["auc"].values
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))
        t_stat = mean_diff / (std_diff / np.sqrt(len(diffs))) if std_diff > 0 else float("inf")
        crit = _t_crit.get(len(diffs), 2.776)
        sig = "显著 ✓" if t_stat >= crit else "n.s."
        print(f"  {mode:<10} Δ={mean_diff:+.4f} ± {std_diff:.4f}  t={t_stat:5.2f} (crit={crit})  {sig}")

    print("\nDone.")


if __name__ == "__main__":
    main()
