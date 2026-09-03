# -*- coding: utf-8 -*-
"""Table: 100k 复刻消融三线表 (test, 3种子 mean±std, 最优加粗)。
数据源: 模型/chain_airseq_100k/multi_seed_results.csv
输出: 绘图/TABLE_ablation_100k.tex
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_CSV = ROOT / "模型" / "chain_airseq_100k" / "multi_seed_results.csv"
OUT = Path(__file__).resolve().parent / "TABLE_ablation_100k.tex"

ROWS = [
    ("airseq_ca_lstm_only", "LSTM (structure only)"),
    ("airseq_ca_gate", "Residual gate"),
    ("airseq_ca_crossattn", "Cross-attention"),
    ("airseq_ca_catgate", "CatGate (ours)"),
]


def fmt(mean, std, best):
    s = f"{mean:.4f}$\\pm${std:.4f}"
    return f"\\textbf{{{s}}}" if best else s


def main():
    df = pd.read_csv(RESULT_CSV)
    te = df[df["split"] == "test"]
    g = te.groupby("model")[["auc", "pr_auc", "f1"]].agg(["mean", "std"])

    best = {m: g[(m, "mean")].max() for m in ("auc", "pr_auc", "f1")}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation of fusion modes on 100k chains (test set, mean$\pm$std over 3 seeds). Best in bold.}",
        r"\label{tab:ablation_100k}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Mode & AUC & PR-AUC & F1 \\",
        r"\midrule",
    ]
    for model, label in ROWS:
        cells = []
        for m in ("auc", "pr_auc", "f1"):
            mean = g.loc[model, (m, "mean")]
            std = g.loc[model, (m, "std")]
            cells.append(fmt(mean, std, abs(mean - best[m]) < 1e-9))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
