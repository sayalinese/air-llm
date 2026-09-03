# -*- coding: utf-8 -*-
"""把检索到的历史案例组织成紧凑的上下文学习文本。"""
import os
import sys

import torch

STAGE2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "阶段2")
sys.path.insert(0, STAGE2_DIR)

from llm_text import load_prompts


LABEL_TEXT = {0: "未延误", 1: "延误"}


def load_retrieval_prompts(csv_path, cache_path, include_similarity=False):
    """返回 {query_sample_id: 检索增强prompt}，不读取query自身标签。"""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Missing retrieval cache: {cache_path}. "
            "Run service/数据创建/5_生成检索增强数据.py first.")

    base_prompts = load_prompts(csv_path)
    cache = torch.load(cache_path, weights_only=False)
    query_ids = torch.as_tensor(cache["query_sample_id"], dtype=torch.long)
    retrieved_ids = torch.as_tensor(cache["retrieved_sample_id"], dtype=torch.long)
    retrieved_labels = torch.as_tensor(cache["retrieved_label"], dtype=torch.long)
    retrieved_days = torch.as_tensor(cache.get("retrieved_day", torch.full_like(retrieved_labels, -1)))
    query_days = torch.as_tensor(cache.get("query_day", torch.zeros(len(query_ids), dtype=torch.long)))
    valid = torch.as_tensor(cache.get("retrieval_valid", retrieved_ids >= 0), dtype=torch.bool)
    similarities = torch.as_tensor(cache["similarity"], dtype=torch.float32)
    prompts = {}

    for row, query_id in enumerate(query_ids.tolist()):
        cases = []
        for rank, example_id in enumerate(retrieved_ids[row].tolist(), 1):
            if not bool(valid[row, rank - 1]) or example_id < 0:
                continue
            example = base_prompts.get(example_id, "")
            label = LABEL_TEXT[int(retrieved_labels[row, rank - 1])]
            score = f"，相似度{similarities[row, rank - 1]:.3f}" if include_similarity else ""
            case_day = int(retrieved_days[row, rank - 1])
            cases.append(f"案例{len(cases) + 1}：{example}，日期4月{case_day}日；结果{label}{score}")
        query = base_prompts.get(query_id, "")
        history = "。".join(cases) if cases else "无可用历史案例"
        query_day = int(query_days[row])
        prompts[query_id] = (
            "历史相似案例：" + history + "。"
            + f"待分类航班：{query}，日期4月{query_day}日。"
            + "预测结果："
        )

    return prompts
