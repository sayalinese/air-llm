"""vLLM 推理: 加载 base model + LoRA adapter → 批量生成"""
import os
import json
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.config import MODEL_PATH, SAVE_DIR, DATA_DIR, PROMPT_TEMPLATE, LABEL_MAP
from service.dataset import build_text_payload


def parse_label(text):
    text = text.strip()
    if text.startswith(LABEL_MAP[1]):
        return 1
    if text.startswith(LABEL_MAP[0]):
        return 0
    return None


def vllm_inference():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    test_path = os.path.join(DATA_DIR, "test.jsonl")
    data = []
    with open(test_path, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    prompts = [PROMPT_TEMPLATE.format(text=build_text_payload(item)) for item in data]
    labels = [item['label'] for item in data]

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        enable_lora=True,
        max_loras=1,
        max_lora_rank=16,
        trust_remote_code=True,
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=5, logprobs=20)
    outputs = llm.generate(
        prompts, sampling,
        lora_request=LoRARequest("delay_lora", 1, SAVE_DIR),
    )

    parsed = [parse_label(out.outputs[0].text) for out in outputs]
    invalid = sum(label is None for label in parsed)
    preds = [0 if label is None else label for label in parsed]

    print(f"Invalid outputs: {invalid}/{len(outputs)}")
    print(f"ACC: {accuracy_score(labels, preds):.4f}")
    print(f"F1:  {f1_score(labels, preds):.4f}")
    print(classification_report(labels, preds, target_names=['正常', '延误'], zero_division=0))
