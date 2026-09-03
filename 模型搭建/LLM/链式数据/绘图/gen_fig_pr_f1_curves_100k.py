# -*- coding: utf-8 -*-
"""Fig: 100k 复刻 PR 曲线 + F1-阈值 全曲线 (双面板, 4 模式)。
数据源: 绘图/predictions_100k_{mode}.csv (test 全量逐样本概率)
输出: 绘图/fig_pr_f1_curves_100k.pdf/.png
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score

from paper_plot_style import plt, save

FIG_DIR = Path(__file__).resolve().parent

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


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax_pr, ax_f1 = axes
    thresholds = np.linspace(0.05, 0.95, 91)
    for mode, label in MODE_LABELS.items():
        df = pd.read_csv(FIG_DIR / f"predictions_100k_{mode}.csv")
        y, p = df["label"].to_numpy(), df["prob"].to_numpy()
        color = MODE_COLORS[mode]
        lw = 1.8 if mode == "catgate" else 1.3
        # PR 曲线
        prec, rec, _ = precision_recall_curve(y, p)
        ax_pr.plot(rec, prec, label=label, color=color, linestyle=MODE_LINESTYLES[mode], linewidth=lw)
        # F1-阈值曲线
        f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in thresholds]
        ax_f1.plot(thresholds, f1s, label=label, color=color, linestyle=MODE_LINESTYLES[mode], linewidth=lw)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0.15, 0.75)
    ax_pr.legend(frameon=False, loc="upper right", fontsize=8)
    ax_f1.set_xlabel("Decision Threshold")
    ax_f1.set_ylabel("F1 Score")
    ax_f1.set_xlim(0.05, 0.95)
    ax_f1.legend(frameon=False, loc="upper right", fontsize=8)
    save(fig, "fig_pr_f1_curves_100k")
    print("Saved: fig_pr_f1_curves_100k")


if __name__ == "__main__":
    main()
