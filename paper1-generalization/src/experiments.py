"""Orchestrate independently implemented Figure 1 experiments.

Runs train.py on a fast local filesystem. Optionally backs up each completed run
(including raw logs) to a persistent directory, such as mounted Google Drive.
No model or training implementation is changed here.

Example (Colab):
    python src/experiments.py --preset full --epochs 100 --device cuda \
      --output-root /content/figure1_runs \
      --backup-root /content/drive/MyDrive/ML-Paper-Reproductions/figure1

Re-running the same command skips verified completed backups. There is no
mid-epoch/epoch checkpointing: interrupted individual jobs restart from scratch.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
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
        return jobs_for("label", models, probabilities) + jobs_for("input", models, probabilities)
    raise ValueError(f"Unknown preset: {preset}")


def directory_for(root: Path, job: Job) -> Path:
    if job.tag == "benchmark":
        return root / "benchmark" / job.model
    if job.tag == "label":
        return root / "label_corruption" / job.model / probability_tag(job.probability)
    return root / "input_randomization" / job.model / job.condition


def status_of(folder: Path) -> str:
    metadata = folder / "metadata.json"
    if not metadata.exists():
        return "partial" if folder.exists() and any(folder.iterdir()) else "new"
    try:
        meta = json.loads(metadata.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return "partial"
    if (meta.get("completed") is True and
            (folder / "epochs.csv").is_file() and
            (folder / "steps.csv").is_file()):
        return "complete"
    return "partial"


def read_last_epoch(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        last = None
        for last in csv.DictReader(f):
            pass
        return last


def summarize(output_root: Path) -> int:
    rows: list[dict] = []
    for meta_path in sorted(output_root.rglob("metadata.json")) if output_root.exists() else []:
        folder = meta_path.parent
        # Ignore abandoned/incomplete archives when building the final summary.
        if any(".incomplete-" in part for part in folder.parts):
            continue
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
        tmp = summary.with_name(summary.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, summary)
    elif summary.exists():
        summary.unlink()
    print(f"Summary: {len(rows)} recorded runs" + (f" -> {summary}" if rows else " (no runs yet)"), flush=True)
    return len(rows)


def expected_config(args: argparse.Namespace, job: Job, epochs: int) -> dict:
    return {
        "model": job.model,
        "condition": job.condition,
        "corruption_probability": job.probability,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "train_subset": args.train_subset,
        "eval_interval": args.eval_interval,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "lr_decay": args.lr_decay,
        "convergence_error": args.convergence_error,
        "convergence_patience": args.convergence_patience,
        "seed": args.seed,
        "label_seed": args.label_seed,
        "input_seed": args.input_seed,
        "num_workers": args.num_workers,
        "data_root": args.data_root,
        "device": args.device,
    }


def matching_complete(folder: Path, expected: dict) -> bool:
    if status_of(folder) != "complete":
        return False
    try:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        cfg = metadata["config"]
        # output_dir is intentionally excluded: local and Drive paths differ.
        if any(cfg.get(key) != value for key, value in expected.items()):
            return False
        last = read_last_epoch(folder / "epochs.csv")
        if not last:
            return False
        final_epoch = int(last["epoch"])
        if expected["max_steps"] is None and final_epoch != expected["epochs"]:
            return False
        if expected["max_steps"] is not None and int(metadata["total_steps"]) != expected["max_steps"]:
            # Training may end naturally before a larger max_steps limit.
            if final_epoch != expected["epochs"]:
                return False
        if int(last["step"]) != int(metadata["total_steps"]):
            return False
        with (folder / "steps.csv").open(newline="", encoding="utf-8") as f:
            step_rows = csv.DictReader(f)
            final_step = None
            for row in step_rows:
                final_step = row.get("step")
        return final_step is not None and int(final_step) == int(metadata["total_steps"])
    except (OSError, ValueError, TypeError, KeyError):
        return False


def archive_partial(folder: Path) -> None:
    if not folder.exists():
        return
    suffix = 1
    while folder.with_name(folder.name + f".incomplete-{suffix}").exists():
        suffix += 1
    archive = folder.with_name(folder.name + f".incomplete-{suffix}")
    folder.rename(archive)
    print(f"  Archived existing local output -> {archive}", flush=True)


def backup_completed(source: Path, destination: Path, expected: dict) -> None:
    """Copy to a unique staging folder; publish only after validating the copy.

    Mounted Drive is not guaranteed to provide POSIX atomic rename semantics.
    Existing valid backups are never overwritten.
    """
    if matching_complete(destination, expected):
        return
    if destination.exists():
        raise RuntimeError(f"Backup destination exists but is incomplete or incompatible: {destination}. "
                           "Move it aside manually or choose another --backup-root.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".uploading-" + uuid.uuid4().hex[:10])
    try:
        shutil.copytree(source, staging)
        if not matching_complete(staging, expected):
            raise RuntimeError(f"Copied backup failed validation: {staging}")
        staging.rename(destination)
        if not matching_complete(destination, expected):
            raise RuntimeError(f"Published backup failed validation: {destination}")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def command_for(args: argparse.Namespace, job: Job, epochs: int, folder: Path) -> list[str]:
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
    for flag, value in (("--max-steps", args.max_steps),
                        ("--train-subset", args.train_subset),
                        ("--learning-rate", args.learning_rate)):
        if value is not None:
            cmd.extend((flag, str(value)))
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent Figure 1 experiment orchestration")
    parser.add_argument("--preset", choices=("benchmark", "label", "input", "full"), default="benchmark")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--probabilities", nargs="+", type=float, default=list(DEFAULT_CORRUPTION))
    parser.add_argument("--epochs", type=int, default=None,
                        help="Default 1 for benchmark, 100 otherwise")
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
    parser.add_argument("--output-root", default="results/figure1",
                        help="Fast LOCAL training output directory (prefer /content on Colab)")
    parser.add_argument("--backup-root", default=None,
                        help="Persistent backup root, e.g. /content/drive/MyDrive/...; copy after each job")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--retry-partial", action="store_true",
                        help="Archive incompatible or incomplete local runs, then retry")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).resolve()
    backup = Path(args.backup_root).resolve() if args.backup_root else None
    if backup and (root == backup or root in backup.parents or backup in root.parents):
        raise SystemExit("--output-root and --backup-root must be separate, non-nested directories")
    if args.summarize_only:
        summarize(backup if backup else root)
        return
    if not args.probabilities or any(not math.isfinite(p) or not 0 <= p <= 1 for p in args.probabilities):
        raise SystemExit("All corruption probabilities must be finite values in [0,1]")
    if len(set(probability_tag(p) for p in args.probabilities)) != len(args.probabilities):
        raise SystemExit("Corruption probabilities collide after rounding to integer percentages")
    if args.epochs is not None and args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    if args.batch_size < 1 or args.eval_interval < 1:
        raise SystemExit("--batch-size and --eval-interval must be positive")
    epochs = args.epochs if args.epochs is not None else (1 if args.preset == "benchmark" else 100)
    jobs = jobs_for(args.preset, args.models, args.probabilities)
    print(f"Preset={args.preset}; jobs={len(jobs)}; epochs={epochs}; batch_size={args.batch_size}; "
          f"subset={args.train_subset or 'full'}; device={args.device}", flush=True)
    print(f"Local output: {root}", flush=True)
    if backup:
        if not args.dry_run:
            # Require the Colab Drive mount itself before creating nested folders.
            if str(backup).startswith("/content/drive/") and not Path("/content/drive/MyDrive").is_dir():
                raise SystemExit("Google Drive is not mounted. Run drive.mount('/content/drive') first")
            backup.mkdir(parents=True, exist_ok=True)
        print(f"Persistent backup: {backup}", flush=True)
    failures = 0
    for index, job in enumerate(jobs, start=1):
        folder = directory_for(root, job)
        destination = directory_for(backup, job) if backup else None
        expected = expected_config(args, job, epochs)
        print(f"[{index}/{len(jobs)}] {job.model} {job.condition} p={job.probability:g} "
              f"-> {folder}", flush=True)
        if destination and matching_complete(destination, expected):
            print(f"  SKIP: verified Drive backup -> {destination}", flush=True)
            continue
        if matching_complete(folder, expected):
            print("  Local run already complete", flush=True)
            if destination and not args.dry_run:
                backup_completed(folder, destination, expected)
                print(f"  BACKED UP -> {destination}", flush=True)
            continue
        state = status_of(folder)
        if folder.exists() and state != "new" and not args.retry_partial:
            print("  SKIP: local output is partial or incompatible; use --retry-partial", flush=True)
            continue
        if destination and destination.exists():
            print(f"  ERROR: incompatible/partial backup exists: {destination}", file=sys.stderr)
            failures += 1
            if not args.continue_on_error:
                break
            continue
        cmd = command_for(args, job, epochs, folder)
        if args.dry_run:
            print("  DRY RUN:", subprocess.list2cmdline(cmd), flush=True)
            continue
        if folder.exists() and state != "new":
            archive_partial(folder)
        try:
            started = time.perf_counter()
            subprocess.run(cmd, cwd=ROOT, check=True)
            if not matching_complete(folder, expected):
                raise RuntimeError(f"Run finished but output validation failed: {folder}")
            print(f"  COMPLETE in {time.perf_counter()-started:.1f}s", flush=True)
            if destination:
                backup_completed(folder, destination, expected)
                print(f"  BACKED UP -> {destination}", flush=True)
        except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break
        finally:
            if backup and backup.exists():
                summarize(backup)
    if not args.dry_run:
        summarize(root)
        if backup:
            summarize(backup)
    if failures:
        raise SystemExit(f"{failures} experiment(s) failed")


if __name__ == "__main__":
    main()
