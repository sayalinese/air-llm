"""训练可视化: 损失曲线 + 精度对比"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .config import SAVE_DIR, DATA_DIR


def plot_loss(history_path=None, save_path=None):
    if history_path is None:
        history_path = os.path.join(SAVE_DIR, 'history.json')
    if save_path is None:
        save_path = os.path.join(SAVE_DIR, 'loss_curve.png')

    with open(history_path) as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, 'o-', label='Train Loss', color='#2196F3', linewidth=2)
    ax.plot(epochs, val_loss, 's--', label='Val Loss', color='#FF5722', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('LoRA Training Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    for e, tl, vl in zip(epochs, train_loss, val_loss):
        ax.annotate(f'{vl:.3f}', (e, vl), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9, color='#FF5722')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_metrics(metrics_path=None, save_path=None):
    """从 evaluate.py 输出的结果画精度对比图"""
    if metrics_path is None:
        metrics_path = os.path.join(SAVE_DIR, 'metrics.json')
    if save_path is None:
        save_path = os.path.join(SAVE_DIR, 'metrics.png')

    if not os.path.exists(metrics_path):
        print(f"metrics.json not found at {metrics_path}")
        print("Run evaluate.py first, or save metrics to metrics.json")
        return

    with open(metrics_path) as f:
        m = json.load(f)

    metrics = m.get('test', m)
    categories = ['ACC', 'AUC', 'PR-AUC', 'F1', 'Precision\n(延误)', 'Recall\n(延误)']
    values = [
        metrics.get('acc') or 0,
        metrics.get('auc') or 0,
        metrics.get('pr_auc') or 0,
        metrics.get('f1') or 0,
        metrics.get('precision') or 0,
        metrics.get('recall') or 0,
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('LoRA Flight Delay Classification Metrics', fontsize=14)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_all():
    plot_loss()
    plot_metrics()


if __name__ == '__main__':
    plot_all()
