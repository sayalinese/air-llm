# -*- coding: utf-8 -*-
"""Fig: 100k 复刻 test AUC 模式对比 (3种子 mean ± std 误差棒)。
数据源: 模型/chain_airseq_100k/multi_seed_results.csv (test 分片)
输出: 绘图/fig_test_auc_compare_100k.pdf/.png
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from paper_plot_style import plt, save

ROOT = Path(__file__).resolve().parents[1]
RESULT_CSV = ROOT / "模型" / "chain_airseq_100k" / "multi_seed_results.csv"

MODE_ORDER = ["lstm_only", "gate", "crossattn", "catgate"]
MODE_LABELS = {
    "lstm_only": "LSTM",
    "gate": "Residual\nGate",
    "crossattn": "Cross-\nAttention",
    "catgate": "CatGate\n(Ours)",
}
MODE_COLORS = {
    "lstm_only": "#7f7f7f",
    "gate": "#9467bd",
    "crossattn": "#1f77b4",
    "catgate": "#d62728",
}


def main():
    df = pd.read_csv(RESULT_CSV)
    te = df[df["split"] == "test"]
    g = te.groupby("model")["auc"].agg(["mean", "std", "count"]).reindex(
        [f"airseq_ca_{m}" for m in MODE_ORDER])

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.0))
    xs = np.arange(len(MODE_ORDER))
    means = g["mean"].to_numpy()
    stds = g["std"].fillna(0).to_numpy()
    bars = ax.bar(xs, means, yerr=stds, capsize=3, width=0.6,
                  color=[MODE_COLORS[m] for m in MODE_ORDER],
                  edgecolor="black", linewidth=0.5)
    for x, m, s, n in zip(xs, means, stds, g["count"].to_numpy()):
        ax.text(x, m + s + 0.002, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text((x + 0.5) / len(xs), 0.03, f"n={int(n)}", ha="center", va="bottom",
                fontsize=7, color="white", transform=ax.transAxes)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], fontsize=8)
    ax.set_ylabel("Test AUC")
    lo = means.min() - stds.max()
    ax.set_ylim(max(0.5, lo - 0.02), means.max() + stds.max() + 0.02)
    save(fig, "fig_test_auc_compare_100k")
    print("Saved: fig_test_auc_compare_100k")


if __name__ == "__main__":
    main()
