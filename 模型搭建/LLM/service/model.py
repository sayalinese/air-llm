"""模型加载: Gemma-4-E4B + LoRA"""
import os
import gc
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Gemma4Config, Gemma4ForCausalLM, Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from .config import (
    MODEL_ARCH,
    MODEL_PATH,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET,
    LORA_TARGET_REGEX,
    SAVE_DIR,
)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_text_from_multimodal(device):
    """只加载多模态 checkpoint 里的文本塔权重。"""
    config = Gemma4Config.from_pretrained(MODEL_PATH)
    model = Gemma4ForCausalLM(config.text_config).to(dtype=torch.bfloat16)
    weights_path = os.path.join(MODEL_PATH, "model.safetensors")
    prefix = "model.language_model."
    target_keys = set(model.state_dict().keys())
    state_dict = {}

    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                target_key = "model." + key[len(prefix):]
                if target_key in target_keys:
                    state_dict[target_key] = f.get_tensor(key)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.tie_weights()
    model.to(device)

    missing = [key for key in missing if key != "lm_head.weight"]
    if missing or unexpected:
        print("Text-only load report:")
        print(f"  missing: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        print(f"  unexpected: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    print(f"Loaded text tower only: {len(state_dict)} tensors")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Has vision_tower: {hasattr(model, 'vision_tower')}")
    print(f"Has audio_tower: {hasattr(model, 'audio_tower')}")
    del state_dict
    gc.collect()
    return model


def load_model(device='cuda'):
    """加载 Gemma-4-E4B。

    默认只加载多模态 checkpoint 中的 language_model 文本塔, 避免视觉/音频塔占显存。
    """
    if MODEL_ARCH == "text_from_multimodal":
        model = _load_text_from_multimodal(device)
        model.config.use_cache = False
        return model

    model_cls = Gemma4ForConditionalGeneration if MODEL_ARCH == "conditional_generation" else Gemma4ForCausalLM
    model = model_cls.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def apply_lora(model):
    target_modules = LORA_TARGET_REGEX if MODEL_ARCH == "conditional_generation" else LORA_TARGET
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def save_lora(model):
    model.save_pretrained(SAVE_DIR)


def load_lora():
    from peft import PeftModel
    base = load_model()
    model = PeftModel.from_pretrained(base, SAVE_DIR)
    return model
