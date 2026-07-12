"""Train selected 10k ablation candidates on val, then evaluate one winner once."""
import argparse
import json
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(HERE, "模型")
STAGES = ("current", "chain", "context", "nearby", "risk")


def experiment_name(stage):
    return f"t60_medium10k_{stage}"


def stage_env(stage):
    env = os.environ.copy()
    env["LLM_ABLATION_STAGE"] = stage
    env["LLM_DATASET_VARIANT"] = "medium"
    env["LLM_EXPERIMENT_NAME"] = experiment_name(stage)
    env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
    return env


def train_and_validate(stage, overwrite=False):
    output_dir = os.path.join(MODEL_ROOT, experiment_name(stage))
    metrics_path = os.path.join(output_dir, "metrics_val.json")
    if os.path.exists(metrics_path) and not overwrite:
        print(f"[{stage}] existing validation metrics found; skipping")
        return metrics_path

    env = stage_env(stage)
    print(f"[{stage}] training 10k -> {output_dir}", flush=True)
    subprocess.run([sys.executable, "train.py"], cwd=HERE, env=env, check=True)
    print(f"[{stage}] validating without test access", flush=True)
    subprocess.run(
        [sys.executable, os.path.join("service", "evaluate.py"), "--val-only"],
        cwd=HERE,
        env=env,
        check=True,
    )
    return metrics_path


def dry_run(stage):
    script = (
        f"import sys; sys.path.insert(0, {HERE!r}); "
        "from service.config import *; "
        "from service.dataset import load_jsonl, encode_prompt; "
        "from service.model import load_tokenizer; "
        "rows=load_jsonl(data_path('train')); tok=load_tokenizer(); "
        "lengths=[len(encode_prompt(tok,x,reserve_tokens=1)) for x in rows]; "
        "print(ABLATION_STAGE, DATASET_VARIANT, len(rows), min(lengths), max(lengths), "
        "SMOKE_NUM_SAMPLES, EPOCHS, SMOKE_NUM_SAMPLES*EPOCHS//GRAD_ACCUM)"
    )
    subprocess.run([sys.executable, "-c", script], cwd=HERE, env=stage_env(stage), check=True)


def summarize(paths):
    rows = []
    for stage in STAGES:
        path = paths.get(stage)
        if path is None or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)
        val = metrics["val"]
        rows.append(
            {
                "stage": stage,
                "val_pr_auc": val["pr_auc"],
                "val_auc": val["auc"],
                "val_f1": val["f1"],
                "threshold": val["threshold"],
            }
        )
    rows.sort(key=lambda row: row["val_pr_auc"], reverse=True)
    summary_path = os.path.join(MODEL_ROOT, "t60_medium10k_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    for row in rows:
        print(
            f"{row['stage']:5s} val_PR={row['val_pr_auc']:.4f} "
            f"val_AUC={row['val_auc']:.4f} val_F1={row['val_f1']:.4f}"
        )
    if rows:
        print(f"Recommended final stage by val PR-AUC: {rows[0]['stage']}")
    print(f"Saved: {summary_path}")


def final_evaluate(stage):
    adapter = os.path.join(MODEL_ROOT, experiment_name(stage), "adapter_config.json")
    if not os.path.exists(adapter):
        raise FileNotFoundError(f"Trained adapter not found: {adapter}")
    print(f"[{stage}] one-time locked final-test evaluation", flush=True)
    subprocess.run(
        [sys.executable, os.path.join("service", "evaluate.py")],
        cwd=HERE,
        env=stage_env(stage),
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Run selected 10k validation experiments.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--final-stage", choices=STAGES, help="Evaluate one selected adapter on locked final test.")
    parser.add_argument("--dry-run", action="store_true", help="Validate medium inputs without loading the model.")
    args = parser.parse_args()
    if args.dry_run:
        for stage in args.stages:
            dry_run(stage)
        return
    if args.final_stage:
        final_evaluate(args.final_stage)
        return

    paths = {stage: train_and_validate(stage, args.overwrite) for stage in args.stages}
    for stage in STAGES:
        existing = os.path.join(MODEL_ROOT, experiment_name(stage), "metrics_val.json")
        if os.path.exists(existing):
            paths.setdefault(stage, existing)
    summarize(paths)


if __name__ == "__main__":
    main()
