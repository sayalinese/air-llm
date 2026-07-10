"""SFT 训练: Gemma + LoRA (支持断点续训)"""
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from service.config import *
from service.dataset import FlightDelayDataset
from service.model import load_tokenizer, load_model, apply_lora, save_lora


CKPT_DIR = os.path.join(SAVE_DIR, 'checkpoint')
MANIFEST_PATH = os.path.join(DATA_DIR, 'llm_dataset_manifest.json')


def load_dataset_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, encoding='utf-8') as f:
        return json.load(f)


def dataset_run_state(manifest):
    return {
        'dataset_schema_version': manifest.get('schema_version'),
        'dataset_prompt_version': manifest.get('prompt_version'),
        'dataset_observation_policy': manifest.get('observation_policy'),
        'dataset_daily_root': manifest.get('daily_root'),
    }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
    }


def build_train_loader(train_ds):
    if TRAIN_SAMPLING_MODE == "balanced_smoke":
        num_samples = SMOKE_NUM_SAMPLES or len(train_ds)
        sampler = WeightedRandomSampler(
            weights=train_ds.sample_weights(target_pos_ratio=BALANCED_POS_RATIO),
            num_samples=num_samples,
            replacement=True,
        )
        print(f"Balanced smoke sampling: {num_samples} samples/epoch, target_pos_ratio={BALANCED_POS_RATIO:.4f}")
        return DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)

    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, history, manifest):
    os.makedirs(CKPT_DIR, exist_ok=True)
    model.save_pretrained(CKPT_DIR)
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        'history': history,
        'train_max_samples': TRAIN_MAX_SAMPLES,
        'val_max_samples': VAL_MAX_SAMPLES,
        'max_len': MAX_LEN,
        'prompt_style': PROMPT_STYLE,
        'sampling_mode': TRAIN_SAMPLING_MODE,
        'balanced_pos_ratio': BALANCED_POS_RATIO,
        'model_arch': MODEL_ARCH,
        **dataset_run_state(manifest),
    }, os.path.join(CKPT_DIR, 'train_state.pt'))
    print(f"  Checkpoint saved (epoch {epoch})")


def load_checkpoint(model, optimizer, scheduler):
    ckpt_path = os.path.join(CKPT_DIR, 'train_state.pt')
    if not os.path.exists(ckpt_path):
        return 0, float('inf'), []
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, CKPT_DIR)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    start_epoch = state['epoch']
    best_loss = state['best_loss']
    history = state['history']
    print(f"Resumed from epoch {start_epoch} (best_val={best_loss:.4f})")
    return start_epoch, best_loss, history, model


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    manifest = load_dataset_manifest()
    if manifest:
        print(
            "Dataset: "
            f"{manifest.get('schema_version')} | "
            f"{manifest.get('prompt_version')} | "
            f"{manifest.get('observation_policy')}"
        )

    tokenizer = load_tokenizer()
    print("Loading model...")
    model = load_model(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    model = apply_lora(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_ds = FlightDelayDataset(os.path.join(DATA_DIR, "train.jsonl"), tokenizer, max_samples=TRAIN_MAX_SAMPLES)
    val_ds = FlightDelayDataset(os.path.join(DATA_DIR, "val.jsonl"), tokenizer, max_samples=VAL_MAX_SAMPLES)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Train labels: {dict(train_ds.label_counts)}, Val labels: {dict(val_ds.label_counts)}")
    print(f"Max samples: train={TRAIN_MAX_SAMPLES}, val={VAL_MAX_SAMPLES}")
    print(f"Prompt style: {PROMPT_STYLE}, Sampling: {TRAIN_SAMPLING_MODE}")

    train_loader = build_train_loader(train_ds)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = max(1, len(train_loader) * EPOCHS // GRAD_ACCUM)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps
    )

    # 断点续训
    ckpt_path = os.path.join(CKPT_DIR, 'train_state.pt')
    start_epoch, best_loss, history = 0, float('inf'), []
    if os.path.exists(ckpt_path):
        from peft import PeftModel
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        same_run = (
            state.get('train_max_samples') == TRAIN_MAX_SAMPLES
            and state.get('val_max_samples') == VAL_MAX_SAMPLES
            and state.get('max_len') == MAX_LEN
            and state.get('prompt_style') == PROMPT_STYLE
            and state.get('sampling_mode') == TRAIN_SAMPLING_MODE
            and state.get('balanced_pos_ratio') == BALANCED_POS_RATIO
            and state.get('model_arch') == MODEL_ARCH
            and all(state.get(k) == v for k, v in dataset_run_state(manifest).items())
        )
        if same_run:
            model = PeftModel.from_pretrained(model, CKPT_DIR, is_trainable=True)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            start_epoch = state['epoch']
            best_loss = state['best_loss']
            history = state['history']
            print(f"Resumed from epoch {start_epoch} (best_val={best_loss:.4f})")
        else:
            print("Checkpoint config differs from current run; starting fresh smoke test")
    else:
        print("Starting fresh training")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM
            pbar.set_postfix(loss=total_loss / (i + 1))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} | train_loss: {total_loss/len(train_loader):.4f} | val_loss: {val_loss:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': total_loss / len(train_loader),
            'val_loss': val_loss,
            'prompt_style': PROMPT_STYLE,
            'sampling_mode': TRAIN_SAMPLING_MODE,
            'balanced_pos_ratio': BALANCED_POS_RATIO,
            'train_max_samples': TRAIN_MAX_SAMPLES,
            'val_max_samples': VAL_MAX_SAMPLES,
            'max_len': MAX_LEN,
            'model_arch': MODEL_ARCH,
            **dataset_run_state(manifest),
        })

        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(SAVE_DIR, exist_ok=True)
            save_lora(model)
            print(f"  Best model saved to {SAVE_DIR}")

        # 每轮存断点
        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_loss, history, manifest)

    # 保存训练历史
    hist_path = os.path.join(SAVE_DIR, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {hist_path}")

    # 自动画损失曲线
    try:
        from service.plot import plot_loss
        plot_loss()
        print("Loss curve generated.")
    except Exception as e:
        print(f"Plot skipped: {e}")

    # 训练完成，清理断点
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("Checkpoint cleaned up (training complete)")

    print("Done!")


if __name__ == '__main__':
    train()
