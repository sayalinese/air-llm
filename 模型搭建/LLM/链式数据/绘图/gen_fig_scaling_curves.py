# -*- coding: utf-8 -*-
"""Fig: 数据规模 scaling 曲线 (5k/10k/20k/100k 链, 种子 42, 同协议)。
数据源: 模型/lstm_temp_filt_{5000,10000,20000,100000}_s42/history.csv
输出: 绘图/fig_scaling_curves.pdf/.png (左: 训练损失, 右: 验证 AUC)
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from paper_plot_style import plt, save

ROOT = Path(__file__).resolve().parents[1]

SCALES = [
    ("5k", ROOT / "模型" / "lstm_temp_filt_5000_s42" / "history.csv", "#9ecae1", "o"),
    ("10k", ROOT / "模型" / "lstm_temp_filt_10000_s42" / "history.csv", "#4292c6", "s"),
    ("20k", ROOT / "模型" / "lstm_temp_filt_20000_s42" / "history.csv", "#2171b5", "^"),
    ("100k", ROOT / "模型" / "lstm_temp_filt_100000_s42" / "history.csv", "#08306b", "D"),
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for label, path, color, marker in SCALES:
        df = pd.read_csv(path)
        axes[0].plot(df["epoch"], df["train_loss"], color=color, marker=marker,
                     markersize=3.5, linewidth=1.4, label=label)
        axes[1].plot(df["epoch"], df["auc"], color=color, marker=marker,
                     markersize=3.5, linewidth=1.4, label=label)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation AUC")
    for ax in axes:
        ax.legend(frameon=False, title="Training chains")
    save(fig, "fig_scaling_curves")
    print("Saved: fig_scaling_curves")


if __name__ == "__main__":
    main()
