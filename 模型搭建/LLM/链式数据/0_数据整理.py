"""0_数据整理 (链式 LSTM 官方口径): 从外层 train/val/test.csv 构建航班链张量。

流程:
  读 train -> 拟合稀疏词表 + dense 均值方差 -> 逐分片构建链张量并缓存到 模型/EXPERIMENT/cache/。
链在各分片内构建 (同一 carrier+flnum+date 不跨分片), 无切分泄露; 实际延误只作标签。
可用环境变量 CHAIN_MAX_ROWS 限制每分片读取行数 (快速冒烟)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from service.config import (
    AIRSEQ_MAX_NEIGHBORS,
    AIRSEQ_WINDOW_MIN,
    LLM_MAX_PREV_LEGS,
    MAX_CHAIN_LEN,
    MAX_TEST_CHAINS,
    MAX_TRAIN_CHAINS,
    MAX_VAL_CHAINS,
    RANDOM_SEED,
    TARGET,
    data_path,
)
from service.chain import build_tensors
from service.dataset import save_encoders, save_split_tensors, save_texts, save_airseq_texts
from service.features import fit_dense_stats, fit_sparse_maps, read_split_frame
from service.llm_text import build_field_map, build_texts
from service.airport_seq import build_airseq_texts

_CAPS = {"train": MAX_TRAIN_CHAINS, "val": MAX_VAL_CHAINS, "test": MAX_TEST_CHAINS}


def main():
    max_rows = os.environ.get("CHAIN_MAX_ROWS")
    max_rows = int(max_rows) if max_rows and max_rows.lower() not in {"none", ""} else None

    print("==================== 0_数据整理 (链式 LSTM) ====================")
    print(f"target={TARGET} | max_chain_len={MAX_CHAIN_LEN} | caps={_CAPS} | max_rows/split={max_rows}")

    print("读取训练集并拟合编码器...")
    train_df = read_split_frame(data_path("train"), max_rows=max_rows)
    sparse_maps, vocab_sizes = fit_sparse_maps(train_df)
    dense_mean, dense_std = fit_dense_stats(train_df)
    save_encoders(sparse_maps, vocab_sizes, dense_mean, dense_std)
    print(f"稀疏词表大小: { {k: v for k, v in vocab_sizes.items()} }")

    for split in ("train", "val", "test"):
        df = train_df if split == "train" else read_split_frame(data_path(split), max_rows=max_rows)
        tensors = build_tensors(
            df, sparse_maps, dense_mean, dense_std,
            max_len=MAX_CHAIN_LEN, max_chains=_CAPS[split], seed=RANDOM_SEED, target=TARGET,
        )
        save_split_tensors(split, tensors)
        print(
            f"[{split}] chains={tensors['num_chains']:,} flights={tensors['num_flights']:,} "
            f"pos_rate={tensors['pos_rate']:.4f} -> cache saved"
        )

        # 链上下文文本 (供 4_链式LLM融合; 仅采样链, 静态无泄露)
        sid = np.asarray(tensors["sample_id"], dtype=object)
        vlen = tensors["valid_len"].numpy()
        needed = {str(sid[i, t]) for i in range(sid.shape[0]) for t in range(int(vlen[i]))}
        field_map = build_field_map(df, needed_ids=needed)
        texts = build_texts(tensors["sample_id"], tensors["valid_len"], field_map, LLM_MAX_PREV_LEGS)
        save_texts(split, texts)
        print(f"[{split}] chain-context texts={len(texts):,} -> saved")

        # 真序列: 机场近窗口计划航班序列 (合规: 仅前序静态, 无实际延误)
        airseq = build_airseq_texts(df, needed, AIRSEQ_WINDOW_MIN, AIRSEQ_MAX_NEIGHBORS)
        save_airseq_texts(split, airseq)
        print(f"[{split}] airport-seq texts={len(airseq):,} -> saved")

    print("\n完成。接着运行 2_基准模型.py (XGB 基线) 与 3_模型训练.py (链式 LSTM)。")


if __name__ == "__main__":
    main()
