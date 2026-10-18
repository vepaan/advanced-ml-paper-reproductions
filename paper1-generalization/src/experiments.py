"""Run independent Figure 1 experiments using the existing train.py harness.

This file does not implement models or training. It launches src/train.py in
separate processes, keeps each run's original logs, and summarizes only actual
completed measurements. No authors' or third-party reproduction code is used.

From the repository root:
    python src/experiments.py --preset benchmark --dry-run
    python src/experiments.py --preset benchmark --models mlp
    python src/experiments.py --preset full --dry-run
    python src/experiments.py --preset full --models mlp --epochs 100
    python src/experiments.py --summarize-only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = Path(__file__).resolve().with_name("train.py")
MODELS = ("mlp", "alexnet", "inception")
INPUT_CONDITIONS = ("shared_shuffle", "independent_shuffle", "gaussian")
DEFAULT_CORRUPTION = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class Job:
    model: str
    condition: str
    probability: float
    tag: str


def probability_tag(value: float) -> str:
    return f"p{round(value * 100):03d}"


def jobs_for(preset: str, models: list[str], probabilities: list[float]) -> list[Job]:
    if preset == "benchmark":
        return [Job(m, "original", 0.0, "benchmark") for m in models]
    if preset == "label":
        return [Job(m, "original", p, "label") for m in models for p in probabilities]
    if preset == "input":
        return [Job(m, c, 0.0, "input") for m in models for c in INPUT_CONDITIONS]
    if preset == "full":
        # Original/unmodified training is already included at corruption p=0.
        return jobs_for("label", models, probabilities) + jobs_for("input", models, probabilities)
    raise ValueError(f"Unknown preset: {preset}")


def directory_for(root: Path, job: Job) -> Path:
    if job.tag == "benchmark":
        return root / "benchmark" / job.model
    if job.tag == "label":
        return root / "label_corruption" / job.model / probability_tag(job.probability)
    return root / "input_randomization" / job.model / job.condition


def status_of(folder: Path) -> str:
    meta_path = folder / "metadata.json"
    if not meta_path.exists():
        if folder.exists() and any(folder.iterdir()):
            return "partial"
        return "new"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return "partial"
    if meta.get("completed") is True and (folder / "epochs.csv").exists() and (folder / "steps.csv").exists():
        return "complete"
    return "partial"


def read_last_epoch(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        last = None
        for last in rows:
            pass
        return last


def summarize(output_root: Path) -> int:
    rows: list[dict] = []
    for meta_path in sorted(output_root.rglob("metadata.json")) if output_root.exists() else []:
        folder = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            print(f"WARNING: invalid metadata: {meta_path}", file=sys.stderr)
            continue
        cfg = meta.get("config", {})
        last = read_last_epoch(folder / "epochs.csv")
        rows.append({
            "run_dir": str(folder.relative_to(output_root)),
            "status": status_of(folder),
            "model": cfg.get("model", ""),
            "condition": cfg.get("condition", ""),
            "corruption_probability": cfg.get("corruption_probability", ""),
            "train_subset": cfg.get("train_subset", ""),
            "epochs_requested": cfg.get("epochs", ""),
            "total_steps": meta.get("total_steps", ""),
            "convergence_detection_step": meta.get("convergence_step", ""),
            "final_epoch": last.get("epoch", "") if last else "",
            "final_train_ce": last.get("train_ce", "") if last else "",
            "final_train_error": last.get("train_error", "") if last else "",
            "final_test_ce": last.get("test_ce", "") if last else "",
            "final_test_error": last.get("test_error", "") if last else "",
            "elapsed_seconds": meta.get("elapsed_seconds", ""),
            "parameter_count": meta.get("parameter_count", ""),
            "observed_label_change_rate": meta.get("observed_label_change_rate", ""),
        })
    output_root.mkdir(parents=True, exist_ok=True)
    summary = output_root / "summary.csv"
    if rows:
        with summary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Summary: {len(rows)} recorded runs" + (f" -> {summary}" if rows else " (no runs yet)"))
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent Figure 1 experiment orchestration")
    parser.add_argument("--preset", choices=("benchmark", "label", "input", "full"), default="benchmark")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--probabilities", nargs="+", type=float, default=list(DEFAULT_CORRUPTION))
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs; default 1 for benchmark, 100 otherwise")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--convergence-error", type=float, default=0.001)
    parser.add_argument("--convergence-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-seed", type=int, default=1234)
    parser.add_argument("--input-seed", type=int, default=5678)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="results/figure1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--retry-partial", action="store_true",
                        help="Archive incomplete runs by renaming their directory, then retry")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    if args.summarize_only:
        summarize(root)
        return
    if not args.probabilities or any(not math.isfinite(p) or not 0 <= p <= 1 for p in args.probabilities):
        raise SystemExit("All corruption probabilities must be finite values in [0,1]")
    if len(set(probability_tag(p) for p in args.probabilities)) != len(args.probabilities):
        raise SystemExit("Corruption probabilities collide after rounding to integer percentages")
    if args.epochs is not None and args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    epochs = args.epochs if args.epochs is not None else (1 if args.preset == "benchmark" else 100)
    jobs = jobs_for(args.preset, args.models, args.probabilities)
    print(f"Preset={args.preset}; jobs={len(jobs)}; epochs={epochs}; batch_size={args.batch_size}; "
          f"subset={args.train_subset or 'full'}; device={args.device}", flush=True)
    failures = 0
    for index, job in enumerate(jobs, start=1):
        folder = directory_for(root, job)
        state = status_of(folder)
        print(f"[{index}/{len(jobs)}] {job.model} {job.condition} p={job.probability:g} "
              f"[{state}] -> {folder}", flush=True)
        if state == "complete":
            print("  SKIP: completed run", flush=True)
            continue
        if state == "partial" and not args.retry_partial:
            print("  SKIP: partial output; inspect it or use --retry-partial", flush=True)
            continue
        cmd = [sys.executable, str(TRAIN_SCRIPT), "--model", job.model,
               "--condition", job.condition, "--corruption-probability", str(job.probability),
               "--epochs", str(epochs), "--batch-size", str(args.batch_size),
               "--eval-interval", str(args.eval_interval),
               "--momentum", str(args.momentum), "--lr-decay", str(args.lr_decay),
               "--convergence-error", str(args.convergence_error),
               "--convergence-patience", str(args.convergence_patience),
               "--seed", str(args.seed), "--label-seed", str(args.label_seed),
               "--input-seed", str(args.input_seed), "--num-workers", str(args.num_workers),
               "--device", args.device, "--data-root", args.data_root,
               "--output-dir", str(folder)]
        if args.max_steps is not None:
            cmd += ["--max-steps", str(args.max_steps)]
        if args.train_subset is not None:
            cmd += ["--train-subset", str(args.train_subset)]
        if args.learning_rate is not None:
            cmd += ["--learning-rate", str(args.learning_rate)]
        if args.dry_run:
            print("  DRY RUN:", subprocess.list2cmdline(cmd), flush=True)
            continue
        if state == "partial":
            suffix = 1
            while folder.with_name(folder.name + f".incomplete-{suffix}").exists():
                suffix += 1
            archive = folder.with_name(folder.name + f".incomplete-{suffix}")
            folder.rename(archive)
            print(f"  Archived partial run -> {archive}", flush=True)
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break
    if not args.dry_run:
        summarize(root)
    if failures:
        raise SystemExit(f"{failures} experiment(s) failed")


if __name__ == "__main__":
    main()
