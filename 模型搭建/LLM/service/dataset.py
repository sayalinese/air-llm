"""JSONL loading and canonical tokenization for flight-delay classification."""
from collections import Counter
import json

import torch
from torch.utils.data import Dataset

from .config import (
    ABLATION_STAGE,
    FAIL_ON_PROMPT_TRUNCATION,
    LABEL_MAP,
    MAX_LEN,
    PROMPT_STYLE,
    PROMPT_TEMPLATE,
)
from .prompt import render_compact_features


def load_jsonl(path, max_samples=None):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
            if max_samples is not None and len(data) >= max_samples:
                break
    return data


def build_text_payload(item, prompt_style=None):
    prompt_style = prompt_style or PROMPT_STYLE
    if prompt_style == "compact" and item.get("features"):
        return render_compact_features(
            item["features"],
            chain_position=item.get("chain_position"),
            chain_valid_length=item.get("chain_valid_length"),
            stage=ABLATION_STAGE,
        )
    if item.get("compact_text"):
        return item["compact_text"].strip()
    if "instruction" in item and "input" in item:
        return f"{item['instruction'].strip()}\n{item['input'].strip()}"
    return item["text"].strip()


def build_prompt(item, prompt_style=None):
    return PROMPT_TEMPLATE.format(text=build_text_payload(item, prompt_style))


def encode_prompt(tokenizer, item, reserve_tokens=0, prompt_style=None):
    prompt_ids = [tokenizer.bos_token_id]
    prompt_ids.extend(tokenizer.encode(build_prompt(item, prompt_style), add_special_tokens=False))
    limit = MAX_LEN - reserve_tokens
    if len(prompt_ids) > limit:
        message = (
            f"Prompt for sample {item.get('sample_id', 'unknown')} has {len(prompt_ids)} tokens; "
            f"limit is {limit}. Shorten the prompt instead of silently truncating it."
        )
        if FAIL_ON_PROMPT_TRUNCATION:
            raise ValueError(message)
        prompt_ids = prompt_ids[:limit]
    return prompt_ids


def validate_label_tokens(tokenizer):
    label_tokens = {
        label: tokenizer.encode(text, add_special_tokens=False)
        for label, text in LABEL_MAP.items()
    }
    invalid = {label: ids for label, ids in label_tokens.items() if len(ids) != 1}
    if invalid:
        raise ValueError(f"All labels must be exactly one token, got: {invalid}")
    return {label: ids[0] for label, ids in label_tokens.items()}


class FlightDelayDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_samples=None, prompt_style=None):
        self.data = load_jsonl(jsonl_path, max_samples=max_samples)
        self.tokenizer = tokenizer
        self.prompt_style = prompt_style or PROMPT_STYLE
        self.label_token_ids = validate_label_tokens(tokenizer)
        self.label_counts = Counter(self.get_label(item) for item in self.data)
        self._encoded_cache = [None] * len(self.data)

    def __len__(self):
        return len(self.data)

    def get_label(self, item):
        if "label" in item:
            return int(item["label"])
        reverse_label = {v: k for k, v in LABEL_MAP.items()}
        return reverse_label[item["output"].strip()]

    def sample_weights(self, target_pos_ratio=None):
        if target_pos_ratio is None:
            return [1.0 / self.label_counts[self.get_label(item)] for item in self.data]

        target_pos_ratio = max(0.0, min(1.0, float(target_pos_ratio)))
        target_ratio = {0: 1.0 - target_pos_ratio, 1: target_pos_ratio}
        return [
            target_ratio[self.get_label(item)] / self.label_counts[self.get_label(item)]
            for item in self.data
        ]

    def __getitem__(self, idx):
        if self._encoded_cache[idx] is not None:
            return self._encoded_cache[idx]

        item = self.data[idx]
        label = self.get_label(item)
        answer_id = self.label_token_ids[label]
        prompt_ids = encode_prompt(
            self.tokenizer,
            item,
            reserve_tokens=1,
            prompt_style=self.prompt_style,
        )
        full_ids = prompt_ids + [answer_id]
        labels = [-100] * len(prompt_ids) + [answer_id]

        pad_len = MAX_LEN - len(full_ids)
        input_ids = [self.tokenizer.pad_token_id] * pad_len + full_ids
        labels = [-100] * pad_len + labels
        attention_mask = [0] * pad_len + [1] * len(full_ids)

        encoded = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        self._encoded_cache[idx] = encoded
        return encoded
