"""数据集: JSONL → tokenize → Dataset"""
from collections import Counter
import json
import torch
from torch.utils.data import Dataset
from .config import PROMPT_TEMPLATE, LABEL_MAP, MAX_LEN, PROMPT_STYLE


def load_jsonl(path, max_samples=None):
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            if max_samples is not None and len(data) >= max_samples:
                break
    return data


def _section_value(text, title):
    marker = f"{title}："
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)

    section_titles = ["当前航班", "前序航班1", "前序航班2", "传播摘要"]
    next_positions = [
        text.find(f"{name}：", start)
        for name in section_titles
        if text.find(f"{name}：", start) >= 0
    ]
    end = min(next_positions) if next_positions else len(text)
    return text[start:end].strip()


def _lines_to_dict(section):
    values = {}
    for line in section.splitlines():
        if "：" not in line:
            continue
        key, value = line.split("：", 1)
        values[key.strip()] = value.strip()
    return values


def _format_fields(title, values, fields):
    parts = []
    for field in fields:
        value = values.get(field)
        if value not in (None, ""):
            parts.append(f"{field}={value}")
    if not parts:
        return f"{title}: 缺失"
    return f"{title}: " + "; ".join(parts)


def build_compact_payload(item):
    raw_text = item.get("input") or item.get("text") or ""
    current = _lines_to_dict(_section_value(raw_text, "当前航班"))
    prev1 = _section_value(raw_text, "前序航班1")
    prev2 = _section_value(raw_text, "前序航班2")
    summary = _lines_to_dict(_section_value(raw_text, "传播摘要"))

    route_fields = ["出发机场", "到达机场", "计划出发时间", "计划到达时间", "航班频次"]
    meta_fields = ["月份", "星期", "承运人", "航班号", "飞机尾号"]
    weather_fields = ["出发地气温", "到达地气温", "出发地降水", "到达地降水", "出发地风速", "到达地风速"]
    summary_fields = [
        "前序出发延误分钟数",
        "前序到达延误分钟数",
        "计划过站缓冲时间(分钟)",
        "前两段出发延误均值",
        "前两段出发延误最大值",
        "前两段出发延误变化趋势",
    ]

    prev1_short = prev1.replace("\n", "; ") if prev1 else "缺失"
    prev2_short = prev2.replace("\n", "; ") if prev2 else "缺失"

    return "\n".join([
        f"链位置: 当前第{item.get('chain_position', '缺失')}段; 链有效长度={item.get('chain_valid_length', '缺失')}",
        _format_fields("当前航班", current, route_fields),
        _format_fields("前序传播摘要", summary, summary_fields),
        f"前序航班1: {prev1_short}",
        f"前序航班2: {prev2_short}",
        _format_fields("天气", current, weather_fields),
        _format_fields("机场与承运人", current, meta_fields),
    ])


def build_text_payload(item, prompt_style=None):
    if item.get("compact_text"):
        return item["compact_text"].strip()

    prompt_style = prompt_style or PROMPT_STYLE
    if prompt_style == "compact":
        return build_compact_payload(item)
    if "instruction" in item and "input" in item:
        instruction = item["instruction"].strip()
        input_text = item["input"].strip()
        return f"{instruction}\n\n{input_text}"
    return item["text"]


class FlightDelayDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_samples=None, prompt_style=None):
        self.data = load_jsonl(jsonl_path, max_samples=max_samples)
        self.tokenizer = tokenizer
        self.prompt_style = prompt_style or PROMPT_STYLE
        self.label_tokens = {
            0: tokenizer.encode(LABEL_MAP[0], add_special_tokens=False),
            1: tokenizer.encode(LABEL_MAP[1], add_special_tokens=False),
        }
        self.label_counts = Counter(self.get_label(item) for item in self.data)

    def __len__(self):
        return len(self.data)

    def get_label(self, item):
        if "label" in item:
            return int(item["label"])
        reverse_label = {v: k for k, v in LABEL_MAP.items()}
        return reverse_label[item["output"].strip()]

    def sample_weights(self, target_pos_ratio=None):
        if target_pos_ratio is None:
            return [
                1.0 / self.label_counts[self.get_label(item)]
                for item in self.data
            ]

        target_pos_ratio = max(0.0, min(1.0, float(target_pos_ratio)))
        target_ratio = {
            0: 1.0 - target_pos_ratio,
            1: target_pos_ratio,
        }
        return [
            target_ratio[self.get_label(item)] / self.label_counts[self.get_label(item)]
            for item in self.data
        ]

    def __getitem__(self, idx):
        item = self.data[idx]
        text_payload = build_text_payload(item, self.prompt_style)
        prompt = PROMPT_TEMPLATE.format(text=text_payload)

        label = self.get_label(item)
        answer = LABEL_MAP[label]

        # 加 BOS token (Gemma 要求)
        prompt_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        max_prompt_len = max(1, MAX_LEN - len(answer_ids))
        prompt_ids = prompt_ids[:max_prompt_len]
        full_ids = prompt_ids + answer_ids

        labels = [-100] * len(prompt_ids) + answer_ids

        # 左 padding (Gemma 要求 padding_side='left')
        pad_len = MAX_LEN - len(full_ids)
        input_ids = [self.tokenizer.pad_token_id] * pad_len + full_ids
        labels = [-100] * pad_len + labels
        attention_mask = [0] * pad_len + [1] * len(full_ids)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }
