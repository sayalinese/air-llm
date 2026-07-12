"""Run the five fixed-probe input ablations sequentially."""
import argparse
import json
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(HERE, "模型")
STAGES = ("current", "chain", "context", "nearby", "risk")
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def experiment_name(stage, micro=False):
    profile = "micro" if micro else "probe"
    return f"t60_{profile}_ablation_{stage}"


def dry_run_stage(stage, micro=False):
    os.environ["LLM_ABLATION_STAGE"] = stage
    os.environ["LLM_EXPERIMENT_NAME"] = experiment_name(stage, micro)
    if micro:
        os.environ["LLM_EPOCHS"] = "1"
        os.environ["LLM_SMOKE_NUM_SAMPLES"] = "512"
    for name in list(sys.modules):
        if name == "service.config" or name == "service.dataset":
            del sys.modules[name]
    from service.config import GRAD_ACCUM, EPOCHS, MAX_LEN, SMOKE_NUM_SAMPLES, data_path
    from service.dataset import encode_prompt, load_jsonl
    from service.model import load_tokenizer

    rows = load_jsonl(data_path("train"))
    tokenizer = load_tokenizer()
    lengths = [len(encode_prompt(tokenizer, item, reserve_tokens=1)) for item in rows]
    updates = SMOKE_NUM_SAMPLES * EPOCHS // GRAD_ACCUM
    print(
        f"[{stage}] rows={len(rows)}, prompt={min(lengths)}..{max(lengths)}/{MAX_LEN - 1}, "
        f"optimizer_updates={updates}"
    )


def run_stage(stage, overwrite=False, micro=False):
    name = experiment_name(stage, micro)
    output_dir = os.path.join(MODEL_ROOT, name)
    metrics_path = os.path.join(output_dir, "metrics_val.json")
    if os.path.exists(metrics_path) and not overwrite:
        print(f"[{stage}] existing metrics found; skipping")
        return metrics_path

    env = os.environ.copy()
    env["LLM_ABLATION_STAGE"] = stage
    env["LLM_EXPERIMENT_NAME"] = name
    if micro:
        env["LLM_EPOCHS"] = "1"
        env["LLM_SMOKE_NUM_SAMPLES"] = "512"
    env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n[{stage}] training -> {output_dir}", flush=True)
    subprocess.run([sys.executable, "train.py"], cwd=HERE, env=env, check=True)
    print(f"[{stage}] evaluating", flush=True)
    subprocess.run(
        [sys.executable, os.path.join("service", "evaluate.py"), "--val-only"],
        cwd=HERE,
        env=env,
        check=True,
    )
    return metrics_path


def write_summary(metrics_paths, micro=False):
    for stage in STAGES:
        path = os.path.join(MODEL_ROOT, experiment_name(stage, micro), "metrics_val.json")
        if os.path.exists(path):
            metrics_paths.setdefault(stage, path)
    rows = []
    for stage in STAGES:
        path = metrics_paths.get(stage)
        if path is None:
            continue
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(
            {
                "stage": stage,
                "experiment_name": metrics["experiment_name"],
                "val_pr_auc": metrics["val"]["pr_auc"],
                "val_auc": metrics["val"]["auc"],
                "val_f1": metrics["val"]["f1"],
                "threshold": metrics["val"]["threshold"],
            }
        )
    profile = "micro" if micro else "probe"
    summary_path = os.path.join(MODEL_ROOT, f"t60_{profile}_ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\nAblation summary:")
    for row in rows:
        print(
            f"{row['stage']:8s} val_PR={row['val_pr_auc']:.4f} "
            f"val_AUC={row['val_auc']:.4f} val_F1={row['val_f1']:.4f}"
        )
    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run fixed-probe LLM input ablations.")
    parser.add_argument("--overwrite", action="store_true", help="Retrain stages that already have metrics.json.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--dry-run", action="store_true", help="Validate all prompts without loading the model.")
    parser.add_argument("--micro", action="store_true", help="Run 512 samples for one epoch in separate output directories.")
    args = parser.parse_args()
    if args.dry_run:
        for stage in args.stages:
            dry_run_stage(stage, micro=args.micro)
        return
    metrics_paths = {
        stage: run_stage(stage, overwrite=args.overwrite, micro=args.micro)
        for stage in args.stages
    }
    write_summary(metrics_paths, micro=args.micro)


if __name__ == "__main__":
    main()
