# -*- coding: utf-8 -*-
"""Fig: 100k 复刻 训练损失迭代曲线 (train loss, 3种子 mean + min-max 带)。
数据源: 模型/chain_airseq_100k/history_airseq_ca_{mode}_s{seed}.csv
输出: 绘图/fig_training_loss_100k.pdf/.png
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from paper_plot_style import plt, save

ROOT = Path(__file__).resolve().parents[1]
HIST_DIR = ROOT / "模型" / "chain_airseq_100k"

MODE_LABELS = {
    "lstm_only": "LSTM (structure only)",
    "gate": "Residual gate",
    "catgate": "CatGate (ours)",
    "crossattn": "Cross-attention",
}
MODE_COLORS = {
    "lstm_only": "#7f7f7f",
    "gate": "#9467bd",
    "catgate": "#d62728",
    "crossattn": "#1f77b4",
}
MODE_LINESTYLES = {
    "lstm_only": "--",
    "gate": "-.",
    "catgate": "-",
    "crossattn": ":",
}


def load_histories(mode):
    frames = []
    for p in sorted(HIST_DIR.glob(f"history_airseq_ca_{mode}_s*.csv")):
        frames.append(pd.read_csv(p))
    return frames


def main():
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.0))
    for mode, label in MODE_LABELS.items():
        frames = load_histories(mode)
        if not frames:
            print(f"[warn] no history for {mode}, skip")
            continue
        df = pd.concat(frames)
        g = df.groupby("epoch")["train_loss"].agg(["mean", "min", "max"]).reset_index()
        color = MODE_COLORS[mode]
        ax.plot(g["epoch"], g["mean"], label=label, color=color,
                linestyle=MODE_LINESTYLES[mode], linewidth=1.6)
        ax.fill_between(g["epoch"], g["min"], g["max"], color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_xlim(0.8, 5.2)   # 截到有效区间, 去掉过拟合尾巴
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig_training_loss_100k")
    print("Saved: fig_training_loss_100k")


if __name__ == "__main__":
    main()
