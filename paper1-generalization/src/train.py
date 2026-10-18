"""Independent training harness for Zhang et al. (2017), Figure 1.

No training is performed on import. Logs are measured, never synthesized.
Architecture fidelity must be reviewed separately before full reproduction.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from data import DataConfig, Figure1CIFAR10, InputCondition, make_loader
from models import MODELS, PAPER_PARAMETER_COUNTS, count_parameters


@dataclass
class TrainConfig:
    model: str = "mlp"
    condition: str = "original"
    corruption_probability: float = 0.0
    batch_size: int = 128
    epochs: int = 1
    max_steps: int | None = None
    train_subset: int | None = None
    eval_interval: int = 1
    learning_rate: float | None = None
    momentum: float = 0.9
    lr_decay: float = 0.95
    convergence_error: float = 0.001
    convergence_patience: int = 2
    stop_on_convergence: bool = False
    seed: int = 42
    label_seed: int = 1234
    input_seed: int = 5678
    num_workers: int = 0
    data_root: str = "data"
    output_dir: str = "results/pilot"
    device: str = "cpu"
    resume_checkpoint: str | None = None
    checkpoint_path: str | None = None
    checkpoint_interval: int = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU deterministic operations; CUDA may require further platform settings.
    torch.use_deterministic_algorithms(True, warn_only=True)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Full-loader mean CE and classification error, evaluated in eval mode."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            total_loss += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return {"loss": total_loss / total, "classification_error": 1 - total_correct / total,
            "correct": total_correct, "samples": total}


def append_csv(path: Path, record: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(record)


def atomic_save_checkpoint(payload: dict, path: Path) -> None:
    """Save a complete epoch-boundary checkpoint (including CSV history)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def checkpoint_identity(config: TrainConfig) -> dict:
    """Properties that must remain identical across resumed training."""
    identity = asdict(config)
    for key in ("epochs", "max_steps", "output_dir", "resume_checkpoint", "checkpoint_path",
                "checkpoint_interval", "stop_on_convergence"):
        identity.pop(key, None)
    return identity


def load_checkpoint(path: Path) -> dict:
    # Checkpoints are pickle-based. Only load files created by this own training code.
    return torch.load(path, map_location="cpu", weights_only=False)


def run(config: TrainConfig) -> None:
    if config.model not in MODELS:
        raise ValueError(f"Unknown model {config.model}")
    if config.checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if config.epochs < 1 or config.batch_size < 1 or config.eval_interval < 1:
        raise ValueError("epochs, batch_size and eval_interval must be positive")
    if config.max_steps is not None and config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.train_subset is not None and not 1 <= config.train_subset <= 50_000:
        raise ValueError("train_subset must be between 1 and 50000")
    if not 0 <= config.corruption_probability <= 1:
        raise ValueError("corruption_probability must be in [0,1]")
    if not 0 <= config.convergence_error <= 1 or config.convergence_patience < 1:
        raise ValueError("Invalid convergence criterion")
    if config.condition != "original" and config.corruption_probability != 0:
        raise ValueError("Run input randomization and label corruption separately")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    set_seed(config.seed)
    device = torch.device(config.device)
    initial_lr = config.learning_rate
    if initial_lr is None:
        initial_lr = 0.1 if config.model == "inception" else 0.01
    if initial_lr <= 0:
        raise ValueError("learning rate must be positive")

    train_ds = Figure1CIFAR10(DataConfig(
        root=config.data_root, condition=InputCondition(config.condition),
        corruption_probability=config.corruption_probability,
        label_seed=config.label_seed, input_seed=config.input_seed))
    test_ds = Figure1CIFAR10(DataConfig(root=config.data_root, train=False))
    # For smoke tests use a fixed subset, sampled without replacement.
    if config.train_subset is not None:
        indices = np.random.default_rng(config.seed).permutation(len(train_ds))[:config.train_subset]
        train_data: Dataset = Subset(train_ds, indices.tolist())
        observed_change = float(np.mean(train_ds.targets[indices] != train_ds.original_targets[indices]))
    else:
        train_data = train_ds
        observed_change = train_ds.realized_label_change_rate

    train_loader = make_loader(train_data, config.batch_size, shuffle=True,
                               num_workers=config.num_workers, seed=config.seed)
    # No shuffle, no dropped samples: whole training split/subset is evaluated.
    train_eval_loader = make_loader(train_data, config.batch_size, shuffle=False,
                                    num_workers=config.num_workers, seed=config.seed)
    test_loader = make_loader(test_ds, config.batch_size, shuffle=False,
                              num_workers=config.num_workers, seed=config.seed)
    model = MODELS[config.model]().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=config.momentum,
                                weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resume = load_checkpoint(Path(config.resume_checkpoint)) if config.resume_checkpoint else None
    if resume is not None:
        if resume["identity"] != checkpoint_identity(config):
            raise ValueError("Checkpoint hyperparameters/data differ from requested run")
        if config.max_steps is not None and resume["step"] >= config.max_steps:
            raise ValueError("New max_steps must exceed checkpoint step")
        if resume["epoch"] >= config.epochs:
            raise ValueError("New epochs budget must exceed checkpoint epoch")
        for name in ("steps.csv", "epochs.csv"):
            (out / name).write_bytes(resume["logs"][name])
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        # Optimizer state can be loaded on CPU then moved to requested device.
        for state in optimizer.state.values():
            for key, val in state.items():
                if torch.is_tensor(val):
                    state[key] = val.to(device)
        scheduler.load_state_dict(resume["scheduler"])
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"])
        if torch.cuda.is_available() and resume.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        train_loader.generator.set_state(resume["loader_rng"])
    else:
        for existing in ("steps.csv", "epochs.csv", "metadata.json"):
            if (out / existing).exists():
                raise FileExistsError(f"{out / existing} already exists; specify a resume checkpoint")
    metadata = {"config": asdict(config), "initial_lr": initial_lr,
                "observed_label_change_rate": observed_change,
                "parameter_count": count_parameters(model),
                "paper_parameter_count": PAPER_PARAMETER_COUNTS[config.model],
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "python_version": sys.version, "utc_start": datetime.now(timezone.utc).isoformat(),
                "convergence_step": None, "completed": False}
    if resume is not None:
        metadata = resume["metadata"]
        metadata["config"] = asdict(config)
        metadata["completed"] = False
        metadata["stop_reason"] = "running"
        metadata["resumed_from_epoch"] = resume["epoch"]
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Model={config.model} params={metadata['parameter_count']:,} "
          f"(paper {metadata['paper_parameter_count']:,}); device={device}; "
          f"train_examples={len(train_data)}; changed_labels={observed_change:.4f}", flush=True)

    global_step = int(resume["step"]) if resume else 0
    consecutive = int(resume["consecutive"]) if resume else 0
    start_epoch = int(resume["epoch"]) + 1 if resume else 1
    previous_elapsed = float(resume.get("elapsed_seconds", 0)) if resume else 0.0
    started = time.perf_counter() - previous_elapsed
    if resume:
        print(f"RESUMED at epoch {resume['epoch']} step {global_step}", flush=True)
    reached_budget = False
    stopped_on_convergence = False
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {global_step + 1}")
            loss.backward()
            optimizer.step()
            global_step += 1
            append_csv(out / "steps.csv", {
                "step": global_step, "epoch": epoch, "batch_size": len(targets),
                "batch_train_ce": float(loss.item()), "lr": optimizer.param_groups[0]["lr"]})
            if config.max_steps is not None and global_step >= config.max_steps:
                reached_budget = True
                break
        # Partial epochs do not advance the LR scheduler.
        if not reached_budget:
            scheduler.step()
        if epoch % config.eval_interval == 0 or reached_budget or epoch == config.epochs:
            train_metrics = evaluate(model, train_eval_loader, device)
            test_metrics = evaluate(model, test_loader, device)
            if train_metrics["classification_error"] <= config.convergence_error:
                consecutive += 1
            else:
                consecutive = 0
            converged = consecutive >= config.convergence_patience
            append_csv(out / "epochs.csv", {
                "epoch": epoch, "step": global_step, "train_ce": train_metrics["loss"],
                "train_error": train_metrics["classification_error"],
                "test_ce": test_metrics["loss"], "test_error": test_metrics["classification_error"],
                "converged": int(converged), "elapsed_seconds": round(time.perf_counter()-started, 3)})
            print(f"epoch={epoch} step={global_step} train_ce={train_metrics['loss']:.5f} "
                  f"train_error={train_metrics['classification_error']:.4%} "
                  f"test_error={test_metrics['classification_error']:.4%} "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
            if converged and metadata["convergence_step"] is None:
                # The actual crossing occurred between evaluations; this is its detection step.
                metadata["convergence_step"] = global_step
            if converged and config.stop_on_convergence:
                stopped_on_convergence = True
                print(f"Early stop: convergence confirmed at epoch {epoch}, step {global_step}", flush=True)
        # Always checkpoint full epochs, including epoch-limited completed runs.
        # An interrupted partial epoch is deliberately replayed from the prior boundary.
        if not reached_budget and config.checkpoint_path and (
            epoch % config.checkpoint_interval == 0 or epoch == config.epochs or stopped_on_convergence
        ):
            metadata["last_checkpoint_epoch"] = epoch
            metadata["last_checkpoint_step"] = global_step
            elapsed = round(time.perf_counter()-started, 3)
            payload = {
                "format_version": 1, "identity": checkpoint_identity(config),
                "epoch": epoch, "step": global_step, "consecutive": consecutive,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loader_rng": train_loader.generator.get_state(),
                "metadata": metadata.copy(), "elapsed_seconds": elapsed,
                "logs": {name: (out / name).read_bytes() for name in ("steps.csv", "epochs.csv")},
            }
            atomic_save_checkpoint(payload, Path(config.checkpoint_path))
            print(f"  CHECKPOINT epoch={epoch} step={global_step} -> {config.checkpoint_path}", flush=True)
        if reached_budget or stopped_on_convergence:
            break
    metadata["completed"] = True
    metadata["total_steps"] = global_step
    metadata["stopped_on_convergence"] = stopped_on_convergence
    metadata["stop_reason"] = ("converged" if stopped_on_convergence else
                               "max_steps" if reached_budget else "epoch_limit")
    metadata["elapsed_seconds"] = round(time.perf_counter()-started, 3)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Finished {global_step} steps in {metadata['elapsed_seconds']}s. "
          f"Convergence detected at: {metadata['convergence_step']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measured CIFAR-10 training run")
    parser.add_argument("--model", choices=list(MODELS), default="mlp")
    parser.add_argument("--condition", choices=[v.value for v in InputCondition], default="original")
    parser.add_argument("--corruption-probability", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-subset", type=int)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--convergence-error", type=float, default=0.001)
    parser.add_argument("--convergence-patience", type=int, default=2)
    parser.add_argument("--stop-on-convergence", action="store_true",
                        help="Stop after convergence criterion is satisfied at evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-seed", type=int, default=1234)
    parser.add_argument("--input-seed", type=int, default=5678)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results/pilot")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    run(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
