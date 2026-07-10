import copy
import json
import math
import os
from dataclasses import dataclass
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


DATA_DIR = r"C:/Users/16960/Desktop/期末论文/模型搭建/data/数据集划分/链数据"
CHAIN_ROOT = r"C:/Users/16960/Desktop/期末论文/三模态数据库建立说明/scripts/Aeolus_V2/dataset/Flight_Chain"
SAVE_DIR = r"C:/Users/16960/Desktop/期末论文/模型搭建/data/保存模型/LSTM"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YEAR = 2024

USE_AMP = DEVICE.type == "cuda"
BATCH_SIZE = 2048
EPOCHS = 30
PATIENCE = 6
NUM_WORKERS = 4
GRAD_CLIP = 1.0
AUTO_REBUILD_SPLITS = True
THRESHOLD_MODE = "precision_floor_recall"
DELAY_PRECISION_FLOOR = 0.50

DENSE_FEATURE_NAMES = [
    "O_TEMP",
    "D_TEMP",
    "O_PRCP",
    "D_PRCP",
    "O_WSPD",
    "D_WSPD",
    "FLIGHTS",
    "PREV_DEP_DELAY",
    "PREV_ARR_DELAY",
    "TURNAROUND_SLACK_MIN",
    "PREV2_DEP_DELAY_MEAN",
    "PREV2_DEP_DELAY_MAX",
    "DEP_DELAY_TREND",
]
SPARSE_FEATURE_NAMES = ["FL_MONTH", "FL_WEEK", "CAH", "CDH", "OI", "DI", "OC_ENC", "FN_ENC", "TE"]

DENSE_FEATURE_NAME_MAP_CN = {
    "O_TEMP": "出发地气温",
    "D_TEMP": "到达地气温",
    "O_PRCP": "出发地降水",
    "D_PRCP": "到达地降水",
    "O_WSPD": "出发地风速",
    "D_WSPD": "到达地风速",
    "FLIGHTS": "航班频次",
    "PREV_DEP_DELAY": "前序出发延误分钟数",
    "PREV_ARR_DELAY": "前序到达延误分钟数",
    "TURNAROUND_SLACK_MIN": "计划过站缓冲时间(分钟)",
    "PREV2_DEP_DELAY_MEAN": "前两段出发延误均值",
    "PREV2_DEP_DELAY_MAX": "前两段出发延误最大值",
    "DEP_DELAY_TREND": "前两段出发延误变化趋势",
}

SPARSE_FEATURE_NAME_MAP_CN = {
    "FL_MONTH": "月份",
    "FL_WEEK": "星期",
    "CAH": "计划到达小时",
    "CDH": "计划出发小时",
    "OI": "出发机场编码",
    "DI": "到达机场编码",
    "OC_ENC": "承运人编码",
    "FN_ENC": "航班号编码",
    "TE": "飞机尾号编码",
}

SEARCH_CONFIGS = [
    {
        "name": "bce_cosine_base",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.30,
        "loss": "bce",
        "scheduler": "cosine",
        "lr": 2e-4,
        "weight_decay": 1e-3,
    },
    {
        "name": "bce_onecycle_wide",
        "hidden": 384,
        "layers": 2,
        "dropout": 0.30,
        "loss": "bce",
        "scheduler": "onecycle",
        "lr": 3e-4,
        "weight_decay": 1e-3,
    },
    {
        "name": "focal_cosine_deep",
        "hidden": 384,
        "layers": 3,
        "dropout": 0.20,
        "loss": "focal",
        "scheduler": "cosine",
        "lr": 2e-4,
        "weight_decay": 1e-3,
    },
]


@dataclass
class SequenceSplit:
    dense: np.ndarray
    sparse: np.ndarray
    labels: np.ndarray
    valid_len: np.ndarray


class BinaryFocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma=2.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        prob = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, prob, 1.0 - prob)
        focal = (1.0 - pt).pow(self.gamma)
        return bce * focal


def split_file_path(name):
    return os.path.join(DATA_DIR, f"chain_{name}.npz")


def current_chain_layout():
    files = sorted(glob(os.path.join(CHAIN_ROOT, str(YEAR), "**", "*.pt"), recursive=True))
    if not files:
        return None
    probe = torch.load(files[0], map_location="cpu", weights_only=True)
    return {
        "dense_dim": int(probe["dense_feat"].shape[-1]),
        "sparse_dim": int(probe["sparse_feat"].shape[-1]),
        "max_chain": int(probe["dense_feat"].shape[1]),
    }


def split_is_ready(name, require_delays=True):
    path = split_file_path(name)
    if not os.path.exists(path):
        return False, f"missing {os.path.basename(path)}"

    data = np.load(path)
    try:
        required = {"dense", "sparse", "labels", "vlen"}
        if not required.issubset(set(data.files)):
            return False, f"{os.path.basename(path)} missing {sorted(required - set(data.files))}"
        if require_delays and "delays" not in data.files:
            return False, f"{os.path.basename(path)} missing delays"

        layout = current_chain_layout()
        if layout is not None:
            if data["dense"].shape[1] != layout["dense_dim"] * layout["max_chain"]:
                return False, f"{os.path.basename(path)} dense layout mismatch"
            if data["sparse"].shape[1] != layout["sparse_dim"] * layout["max_chain"]:
                return False, f"{os.path.basename(path)} sparse layout mismatch"
        return True, "ok"
    finally:
        data.close()


def valid_mask(lengths, max_chain):
    steps = np.arange(max_chain)[None, :]
    return steps < lengths[:, None]


def rebuild_chain_splits():
    files = sorted(glob(os.path.join(CHAIN_ROOT, str(YEAR), "**", "*.pt"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No Flight_Chain pt files found under {CHAIN_ROOT}/{YEAR}")

    dates = [os.path.basename(path).replace("flight_chain_", "").replace(".pt", "") for path in files]
    unique_dates = sorted(set(dates))
    c1, c2 = int(len(unique_dates) * 0.6), int(len(unique_dates) * 0.8)
    splits = {
        "train": set(unique_dates[:c1]),
        "val": set(unique_dates[c1:c2]),
        "test": set(unique_dates[c2:]),
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    probe = torch.load(files[0], map_location="cpu", weights_only=True)
    info = {
        "dense_dim": int(probe["dense_feat"].shape[-1]),
        "sparse_dim": int(probe["sparse_feat"].shape[-1]),
        "max_chain": int(probe["dense_feat"].shape[1]),
        "dense_names": probe.get("dense_names", DENSE_FEATURE_NAMES),
        "dense_names_cn": probe.get(
            "dense_names_cn",
            [DENSE_FEATURE_NAME_MAP_CN[name] for name in probe.get("dense_names", DENSE_FEATURE_NAMES)],
        ),
        "sparse_names": probe.get("sparse_names", SPARSE_FEATURE_NAMES),
        "sparse_names_cn": probe.get(
            "sparse_names_cn",
            [SPARSE_FEATURE_NAME_MAP_CN[name] for name in probe.get("sparse_names", SPARSE_FEATURE_NAMES)],
        ),
        "split_stats": {},
    }

    for name in ["train", "val", "test"]:
        dense_parts, sparse_parts, label_parts, vlen_parts, delay_parts = [], [], [], [], []
        rows = 0
        for path in files:
            ymd = os.path.basename(path).replace("flight_chain_", "").replace(".pt", "")
            if ymd not in splits[name]:
                continue

            payload = torch.load(path, map_location="cpu", weights_only=True)
            dense = payload["dense_feat"].numpy()
            sparse = payload["sparse_feat"].numpy()
            labels = payload["labels"].numpy()
            vlen = payload["valid_len"].numpy()
            delays = payload["delays"].numpy()

            dense_parts.append(dense.reshape(dense.shape[0], -1))
            sparse_parts.append(sparse.reshape(sparse.shape[0], -1))
            label_parts.append(labels.reshape(labels.shape[0], -1))
            vlen_parts.append(vlen)
            delay_parts.append(delays.reshape(delays.shape[0], -1))
            rows += dense.shape[0]

        if rows == 0:
            raise ValueError(f"Split {name} has no rows")

        dense = np.concatenate(dense_parts, axis=0)
        sparse = np.concatenate(sparse_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0)
        vlen = np.concatenate(vlen_parts, axis=0)
        delays = np.concatenate(delay_parts, axis=0)

        np.savez(
            split_file_path(name),
            dense=dense,
            sparse=sparse,
            labels=labels,
            vlen=vlen,
            delays=delays,
        )

        mask = valid_mask(vlen, info["max_chain"])
        info["split_stats"][name] = {
            "rows": int(rows),
            "positive_rate": float(labels[mask].mean()),
            "avg_valid_len": float(vlen.mean()),
        }
        print(f"Rebuilt {name}: {info['split_stats'][name]}")

    with open(os.path.join(DATA_DIR, "chain_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def ensure_chain_splits(require_delays=True):
    statuses = {name: split_is_ready(name, require_delays=require_delays) for name in ["train", "val", "test"]}
    missing = {name: reason for name, (ok, reason) in statuses.items() if not ok}
    layout = current_chain_layout()
    if layout is not None and layout["dense_dim"] != len(DENSE_FEATURE_NAMES):
        raise RuntimeError(
            "Flight_Chain raw .pt files are still using the old dense layout. "
            "Please rerun 6_build_chain_fast.py for 2024 with --overwrite before training."
        )
    info_path = os.path.join(DATA_DIR, "chain_info.json")
    if not os.path.exists(info_path):
        missing["chain_info"] = "missing chain_info.json"
    else:
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        if layout is not None:
            if (
                int(info.get("dense_dim", -1)) != layout["dense_dim"]
                or int(info.get("sparse_dim", -1)) != layout["sparse_dim"]
                or int(info.get("max_chain", -1)) != layout["max_chain"]
            ):
                missing["chain_info"] = "chain_info layout mismatch"

    if not missing:
        return

    print("Chain splits are incomplete or stale:")
    for name, reason in missing.items():
        print(f"  - {name}: {reason}")

    if not AUTO_REBUILD_SPLITS:
        raise RuntimeError("Chain splits are incomplete. Rebuild them before training.")

    print("Auto rebuilding chain_train/val/test.npz ...")
    rebuild_chain_splits()


def load_split(name, max_chain, dense_dim, sparse_dim):
    data = np.load(split_file_path(name))
    try:
        dense = data["dense"].reshape(-1, max_chain, dense_dim).astype(np.float32)
        sparse = np.maximum(data["sparse"].reshape(-1, max_chain, sparse_dim).astype(np.int64), 0)
        labels = data["labels"].reshape(-1, max_chain).astype(np.float32)
        valid_len = data["vlen"].astype(np.int64)
        return SequenceSplit(dense=dense, sparse=sparse, labels=labels, valid_len=valid_len)
    finally:
        data.close()


def fit_dense_normalizer(train_split):
    mask = valid_mask(train_split.valid_len, train_split.dense.shape[1])
    feat_dim = train_split.dense.shape[-1]
    valid_values = train_split.dense[mask].reshape(-1, feat_dim)
    mean = valid_values.mean(axis=0).astype(np.float32)
    std = valid_values.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_dense_normalizer(split, mean, std):
    split.dense = ((split.dense - mean) / std).astype(np.float32)
    return split


class ChainDataset(Dataset):
    def __init__(self, split):
        self.dense = torch.from_numpy(split.dense)
        self.sparse = torch.from_numpy(split.sparse)
        self.labels = torch.from_numpy(split.labels)
        self.valid_len = torch.from_numpy(split.valid_len)

    def __len__(self):
        return self.dense.shape[0]

    def __getitem__(self, idx):
        return self.dense[idx], self.sparse[idx], self.labels[idx], self.valid_len[idx]


class ChainLSTM(nn.Module):
    def __init__(self, dense_dim, emb_dims, hidden=256, n_layers=2, drop=0.3):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, d, padding_idx=0) for n, d in emb_dims])
        emb_out = sum(d for _, d in emb_dims)
        input_dim = dense_dim + emb_out

        self.in_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=drop if n_layers > 1 else 0.0,
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, 1),
        )

    def forward(self, dense, sparse, valid_len):
        emb = torch.cat([layer(sparse[:, :, i]) for i, layer in enumerate(self.embs)], dim=-1)
        x = self.in_norm(torch.cat([dense, emb], dim=-1))
        packed = pack_padded_sequence(x, lengths=valid_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=dense.shape[1])
        out = self.out_norm(out)
        return self.head(out).squeeze(-1)


def make_loader(split, shuffle):
    dataset = ChainDataset(split)
    use_workers = NUM_WORKERS if DEVICE.type == "cuda" else 0
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=use_workers,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=use_workers > 0,
    )


def compute_pos_weight(labels, valid_len):
    mask = valid_mask(valid_len, labels.shape[1])
    y = labels[mask]
    pos = float(y.sum())
    neg = float(y.size - pos)
    return neg / max(pos, 1.0)


def compute_threshold_metrics(trues, pred):
    pred = pred.astype(np.int32)
    trues = trues.astype(np.int32)
    tp = int(((pred == 1) & (trues == 1)).sum())
    tn = int(((pred == 0) & (trues == 0)).sum())
    fp = int(((pred == 1) & (trues == 0)).sum())
    fn = int(((pred == 0) & (trues == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    bal_acc = 0.5 * (recall + specificity)
    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_acc": float(bal_acc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_thresholds(probs, trues):
    best = {
        "f1": {"threshold": 0.5, "score": -1.0},
        "delay_precision": {"threshold": 0.5, "score": -1.0},
        "balanced_acc": {"threshold": 0.5, "score": -1.0},
        "precision_floor_recall": {"threshold": 0.5, "score": -1.0},
    }
    floor_candidates = []
    fallback_candidates = []

    for t in np.linspace(0.1, 0.9, 81):
        pred = (probs >= t).astype(np.int32)
        metrics = compute_threshold_metrics(trues, pred)
        metrics["threshold"] = float(t)

        if metrics["f1"] > best["f1"]["score"]:
            best["f1"] = {"threshold": float(t), "score": metrics["f1"], "metrics": metrics}
        if (
            metrics["precision"] > best["delay_precision"]["score"]
            or (
                metrics["precision"] == best["delay_precision"]["score"]
                and metrics["recall"] > best["delay_precision"].get("metrics", {}).get("recall", -1.0)
            )
        ):
            best["delay_precision"] = {"threshold": float(t), "score": metrics["precision"], "metrics": metrics}
        if metrics["balanced_acc"] > best["balanced_acc"]["score"]:
            best["balanced_acc"] = {"threshold": float(t), "score": metrics["balanced_acc"], "metrics": metrics}

        floor_gap = abs(metrics["precision"] - DELAY_PRECISION_FLOOR)
        fallback_candidates.append((floor_gap, -metrics["recall"], -metrics["f1"], float(t), metrics))
        if metrics["precision"] >= DELAY_PRECISION_FLOOR:
            floor_candidates.append((-metrics["recall"], -metrics["f1"], float(t), metrics))

    if floor_candidates:
        _, _, threshold, metrics = min(floor_candidates)
        metrics = dict(metrics)
        metrics["meets_floor"] = True
        best["precision_floor_recall"] = {"threshold": threshold, "score": metrics["recall"], "metrics": metrics}
    else:
        _, _, _, threshold, metrics = min(fallback_candidates)
        metrics = dict(metrics)
        metrics["meets_floor"] = False
        best["precision_floor_recall"] = {"threshold": threshold, "score": metrics["recall"], "metrics": metrics}

    return best


def build_loss(config, pos_weight_value):
    effective_pos_weight = pos_weight_value ** 0.5
    if config["loss"] == "focal":
        return BinaryFocalLoss(effective_pos_weight, gamma=2.0).to(DEVICE)
    return nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(effective_pos_weight, dtype=torch.float32, device=DEVICE),
    )


def build_scheduler(config, optimizer, steps_per_epoch):
    if config["scheduler"] == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=50.0,
        )
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=6,
        T_mult=2,
        eta_min=1e-5,
    )


def evaluate(model, loader, criterion, max_chain):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_prob, all_true = [], []

    with torch.no_grad():
        for dense, sparse, labels, vlen in loader:
            dense = dense.to(DEVICE, non_blocking=True)
            sparse = sparse.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            vlen = vlen.to(DEVICE, non_blocking=True)
            mask = torch.arange(max_chain, device=DEVICE).unsqueeze(0) < vlen.unsqueeze(1)

            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(dense, sparse, vlen)
                loss = (criterion(logits, labels) * mask.float()).sum() / mask.sum()

            prob = torch.sigmoid(logits)
            total_loss += loss.item() * int(mask.sum().item())
            total_tokens += int(mask.sum().item())

            mask_cpu = mask.cpu()
            all_prob.append(prob.cpu()[mask_cpu].numpy())
            all_true.append(labels.cpu()[mask_cpu].numpy())

    probs = np.concatenate(all_prob)
    trues = np.concatenate(all_true)
    return total_loss / max(total_tokens, 1), probs, trues


def checkpoint_score(val_loss, val_auc, threshold_candidates):
    pfr = threshold_candidates["precision_floor_recall"]["metrics"]
    meets_floor = 1 if pfr.get("meets_floor", False) else 0
    return (
        meets_floor,
        pfr["recall"],
        pfr["f1"],
        val_auc,
        -val_loss,
    )


def train_one_experiment(config, train_loader, val_loader, max_chain, actual_dense_dim, emb_dims, pos_weight_value):
    model = ChainLSTM(
        actual_dense_dim,
        emb_dims,
        hidden=config["hidden"],
        n_layers=config["layers"],
        drop=config["dropout"],
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = build_scheduler(config, optimizer, len(train_loader))
    criterion = build_loss(config, pos_weight_value)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    best_state = None
    best_summary = None
    best_score = None
    wait = 0

    print(f"\n=== Experiment: {config['name']} ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total = 0.0
        train_tokens = 0

        for dense, sparse, labels, vlen in train_loader:
            dense = dense.to(DEVICE, non_blocking=True)
            sparse = sparse.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            vlen = vlen.to(DEVICE, non_blocking=True)
            mask = torch.arange(max_chain, device=DEVICE).unsqueeze(0) < vlen.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(dense, sparse, vlen)
                loss = (criterion(logits, labels) * mask.float()).sum() / mask.sum()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            if config["scheduler"] == "onecycle":
                scheduler.step()

            train_total += loss.item() * int(mask.sum().item())
            train_tokens += int(mask.sum().item())

        if config["scheduler"] != "onecycle":
            scheduler.step()

        train_loss = train_total / max(train_tokens, 1)
        val_loss, val_prob, val_true = evaluate(model, val_loader, criterion, max_chain)
        val_auc = float(roc_auc_score(val_true, val_prob))
        threshold_candidates = find_best_thresholds(val_prob, val_true)
        score = checkpoint_score(val_loss, val_auc, threshold_candidates)
        pfr = threshold_candidates["precision_floor_recall"]["metrics"]
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f} | pfr_precision={pfr['precision']:.4f} | pfr_recall={pfr['recall']:.4f}"
        )

        if best_score is None or score > best_score:
            best_score = score
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
            best_summary = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "threshold_candidates": threshold_candidates,
                "selection_score": list(score),
            }
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_summary


def summarize_mode_result(mode_name, threshold, probs, trues):
    pred = (probs >= threshold).astype(np.int32)
    metrics = compute_threshold_metrics(trues, pred)
    metrics["threshold"] = float(threshold)
    metrics["auc"] = float(roc_auc_score(trues, probs))
    return pred, metrics


def main():
    torch.backends.cudnn.benchmark = DEVICE.type == "cuda"
    os.makedirs(SAVE_DIR, exist_ok=True)
    ensure_chain_splits(require_delays=True)

    with open(os.path.join(DATA_DIR, "chain_info.json"), encoding="utf-8") as f:
        info = json.load(f)

    max_chain = int(info["max_chain"])
    dense_dim = int(info["dense_dim"])
    sparse_dim = int(info["sparse_dim"])

    train_split = load_split("train", max_chain, dense_dim, sparse_dim)
    val_split = load_split("val", max_chain, dense_dim, sparse_dim)
    test_split = load_split("test", max_chain, dense_dim, sparse_dim)

    dense_mean, dense_std = fit_dense_normalizer(train_split)
    train_split = apply_dense_normalizer(train_split, dense_mean, dense_std)
    val_split = apply_dense_normalizer(val_split, dense_mean, dense_std)
    test_split = apply_dense_normalizer(test_split, dense_mean, dense_std)

    actual_dense_dim = train_split.dense.shape[-1]
    sparse_max = np.maximum.reduce(
        [
            train_split.sparse.max(axis=(0, 1)),
            val_split.sparse.max(axis=(0, 1)),
            test_split.sparse.max(axis=(0, 1)),
        ]
    ) + 1
    emb_dims = [(int(n), min(max(int(math.sqrt(max(n, 2))) * 2, 4), 64)) for n in sparse_max]
    pos_weight_value = compute_pos_weight(train_split.labels, train_split.valid_len)

    print(f"device: {DEVICE}")
    print(f"dense_dim: {actual_dense_dim}, sparse_dim: {sparse_dim}, max_chain: {max_chain}")
    print(f"train positive rate: {train_split.labels[valid_mask(train_split.valid_len, max_chain)].mean():.4f}")
    print(f"pos_weight: {pos_weight_value:.4f}")
    print(f"embedding dims: {emb_dims}")

    train_loader = make_loader(train_split, shuffle=True)
    val_loader = make_loader(val_split, shuffle=False)
    test_loader = make_loader(test_split, shuffle=False)

    experiment_results = []
    for config in SEARCH_CONFIGS:
        model, summary = train_one_experiment(
            config,
            train_loader,
            val_loader,
            max_chain,
            actual_dense_dim,
            emb_dims,
            pos_weight_value,
        )
        experiment_results.append({"config": config, "model": model, "summary": summary})

    best_experiment = max(experiment_results, key=lambda item: tuple(item["summary"]["selection_score"]))
    best_model = best_experiment["model"]
    best_config = best_experiment["config"]
    best_summary = best_experiment["summary"]
    print(f"\nSelected best experiment: {best_config['name']}")

    val_loss, val_prob, val_true = evaluate(best_model, val_loader, build_loss(best_config, pos_weight_value), max_chain)
    threshold_candidates = best_summary["threshold_candidates"]
    print("validation threshold candidates:")
    for mode_name in ["f1", "balanced_acc", "precision_floor_recall", "delay_precision"]:
        item = threshold_candidates[mode_name]
        metrics = item["metrics"]
        extra = ""
        if mode_name == "precision_floor_recall":
            extra = f", meets_floor={metrics.get('meets_floor', False)}"
        print(
            f"  {mode_name}: t={item['threshold']:.2f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"bal_acc={metrics['balanced_acc']:.4f}, acc={metrics['acc']:.4f}{extra}"
        )

    selected_threshold = threshold_candidates[THRESHOLD_MODE]
    print(f"selected threshold mode: {THRESHOLD_MODE}, threshold={selected_threshold['threshold']:.2f}")

    test_loss, test_prob, test_true = evaluate(best_model, test_loader, build_loss(best_config, pos_weight_value), max_chain)
    test_mode_reports = {}
    for mode_name in ["f1", "balanced_acc", "precision_floor_recall", "delay_precision"]:
        threshold = threshold_candidates[mode_name]["threshold"]
        pred, metrics = summarize_mode_result(mode_name, threshold, test_prob, test_true)
        test_mode_reports[mode_name] = metrics
        print(
            f"test[{mode_name}] t={threshold:.2f} | precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | f1={metrics['f1']:.4f} | "
            f"bal_acc={metrics['balanced_acc']:.4f} | acc={metrics['acc']:.4f} | auc={metrics['auc']:.4f}"
        )

    final_pred = (test_prob >= selected_threshold["threshold"]).astype(np.int32)
    print(f"\nTEST loss: {test_loss:.4f}")
    print(f"ACC: {accuracy_score(test_true, final_pred):.4f}")
    print(f"AUC: {roc_auc_score(test_true, test_prob):.4f}")
    print(f"F1:  {f1_score(test_true, final_pred, zero_division=0):.4f}")
    print(classification_report(test_true, final_pred, target_names=["正常", "延误"], zero_division=0))

    best_path = os.path.join(SAVE_DIR, "chain_lstm.pth")
    torch.save(best_model.state_dict(), best_path)

    meta = {
        "best_config": best_config,
        "best_summary": best_summary,
        "all_experiments": [
            {"config": item["config"], "summary": item["summary"]}
            for item in experiment_results
        ],
        "dense_dim": actual_dense_dim,
        "sparse_dim": sparse_dim,
        "max_chain": max_chain,
        "dense_names": info.get("dense_names", DENSE_FEATURE_NAMES),
        "dense_names_cn": info.get(
            "dense_names_cn",
            [DENSE_FEATURE_NAME_MAP_CN[name] for name in info.get("dense_names", DENSE_FEATURE_NAMES)],
        ),
        "sparse_names": info.get("sparse_names", SPARSE_FEATURE_NAMES),
        "sparse_names_cn": info.get(
            "sparse_names_cn",
            [SPARSE_FEATURE_NAME_MAP_CN[name] for name in info.get("sparse_names", SPARSE_FEATURE_NAMES)],
        ),
        "emb_dims": emb_dims,
        "pos_weight_raw": float(pos_weight_value),
        "pos_weight_used": float(pos_weight_value ** 0.5),
        "threshold_mode": THRESHOLD_MODE,
        "delay_precision_floor": DELAY_PRECISION_FLOOR,
        "threshold_candidates": threshold_candidates,
        "test_mode_reports": test_mode_reports,
        "feature_note": {
            "dense_features_include_missing_mask": False,
            "dense_features_include_prev_arr_delay": True,
            "dense_features_include_turnaround_slack": True,
            "dense_features_include_prev2_stats": True,
            "dense_standardized_on_train_only": True,
            "causal_lstm": True,
        },
        "dense_mean": dense_mean.tolist(),
        "dense_std": dense_std.tolist(),
    }
    with open(os.path.join(SAVE_DIR, "chain_lstm_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
