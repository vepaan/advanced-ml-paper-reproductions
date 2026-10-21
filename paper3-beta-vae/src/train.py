"""Train either beta=1 or beta=4. Resume-safe, offline dSprites, lean checkpoints."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from .data import DSprites, DEFAULT_DATA, N_IMAGES, batches
from .models import BetaVAE, ModelConfig, loss_terms, model_metadata


def tag(beta: float) -> str:
    return f'beta{beta:g}'.replace('.', 'p')


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_atomic(state: dict, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, dest)


def _backup(source: Path, backup_dir: Path | None):
    if backup_dir is None:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    tmp = backup_dir / (source.name + '.tmp')
    shutil.copy2(source, tmp)
    os.replace(tmp, backup_dir / source.name)


def _rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state.get('cuda') is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def train(args):
    if args.beta not in (1, 4):
        raise ValueError('Figure 7A/B only requires beta=1 and beta=4')
    if args.max_steps <= 0 or args.batch_size <= 0:
        raise ValueError('max-steps and batch-size must be positive')
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    ds = DSprites(args.data, args.cache_dir)
    config = ModelConfig()
    model = BetaVAE(config).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
    run_dir = args.output_root / tag(args.beta)
    run_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = args.backup_root / tag(args.beta) if args.backup_root else None
    local_ckpt = run_dir / 'latest.pt'
    backup_ckpt = backup_dir / 'latest.pt' if backup_dir else None
    start_epoch = 0
    start_batch = 0
    global_step = 0
    history: list[dict] = []
    # If neither ckpt exists, identical seed -> identical initialization for both beta arms.
    source = local_ckpt if local_ckpt.is_file() else backup_ckpt if backup_ckpt and backup_ckpt.is_file() else None
    if source:
        ckpt = torch.load(source, map_location=device, weights_only=False)
        expected = {'beta': args.beta, 'seed': args.seed, 'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate, 'model_config': model_metadata(config), 'dataset_length': N_IMAGES}
        for key, value in expected.items():
            if ckpt[key] != value:
                raise ValueError(f'Resume mismatch for {key}: checkpoint={ckpt[key]}, requested={value}')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        _restore_rng(ckpt['rng'])
        start_epoch = int(ckpt['epoch'])
        start_batch = int(ckpt['next_batch'])
        global_step = int(ckpt['step'])
        history = ckpt.get('history', [])
        print(f'RESUMED {tag(args.beta)} at step {global_step} (epoch {start_epoch}, next batch {start_batch}) from {source}', flush=True)
    else:
        print(f'NEW run {tag(args.beta)}: seed {args.seed}, Adagrad lr={args.learning_rate}; '
              'identical initialization across beta arms when seeds match', flush=True)

    print(f'Parameters={sum(p.numel() for p in model.parameters()):,}; samples={len(ds.images):,}; '
          f'batch={args.batch_size}; requested_steps={args.max_steps}; device={device}', flush=True)
    t0 = time.monotonic()
    per_epoch = math.ceil(N_IMAGES / args.batch_size)

    def persist(epoch, next_batch):
        state = {
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'model_config': model_metadata(config), 'beta': args.beta,
            'seed': args.seed, 'batch_size': args.batch_size,
            'learning_rate': args.learning_rate, 'dataset_length': N_IMAGES,
            'step': global_step, 'epoch': epoch, 'next_batch': next_batch,
            'history': history, 'rng': _rng_state(),
        }
        _save_atomic(state, local_ckpt)
        _backup(local_ckpt, backup_dir)
        infer = {k: state[k] for k in ('model','model_config','beta','seed','step')}
        _save_atomic(infer, run_dir / 'model.pt')
        _backup(run_dir / 'model.pt', backup_dir)
        (run_dir / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'SAVED step={global_step}: {local_ckpt}', flush=True)

    if global_step >= args.max_steps:
        print(f'Already at step {global_step} >= requested {args.max_steps}; skipping training.')
        # ensure inference checkpoint exists on local disk even if restored from backup
        persist(start_epoch, start_batch)
        return

    model.train()
    # Accumulate tensors on GPU; sync only every log_every steps.
    metrics = torch.zeros(3, device=device)
    window_steps = 0
    epoch = start_epoch
    while global_step < args.max_steps:
        for bi, indices in batches(N_IMAGES, args.batch_size, seed=args.seed, epoch=epoch,
                                   start_batch=start_batch if epoch == start_epoch else 0):
            x = ds.batch(indices).to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model(x)
            loss, recon, kl = loss_terms(logits, x, mu, logvar, args.beta)
            loss.backward()
            optimizer.step()
            metrics += torch.stack((loss.detach(), recon.detach(), kl.detach()))
            window_steps += 1
            global_step += 1
            next_epoch, next_batch = epoch, bi + 1
            if next_batch == per_epoch:
                next_epoch, next_batch = epoch + 1, 0
            if global_step % args.log_every == 0 or global_step == args.max_steps:
                if not torch.isfinite(metrics).all().item():
                    raise FloatingPointError(f'Non-finite accumulated metrics near step {global_step}')
                values = (metrics / window_steps).cpu().tolist()
                row = {'step': global_step, 'epoch': round(global_step / per_epoch, 4),
                       'loss': round(values[0], 5), 'recon': round(values[1], 5),
                       'kl': round(values[2], 5), 'elapsed_sec': round(time.monotonic()-t0, 1)}
                history.append(row)
                print(json.dumps(row), flush=True)
                metrics.zero_()
                window_steps = 0
            if global_step % args.ckpt_every == 0 or global_step == args.max_steps:
                persist(next_epoch, next_batch)
            if global_step >= args.max_steps:
                break
        epoch += 1
    print(f'COMPLETE {tag(args.beta)} steps={global_step} elapsed={time.monotonic()-t0:.1f}s', flush=True)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--beta', type=float, required=True, choices=[1.0, 4.0])
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--output-root', type=Path, default=Path('/content/paper3_runs' if Path('/content').exists() else './results'))
    p.add_argument('--backup-root', type=Path, default=None)
    p.add_argument('--max-steps', type=int, default=12000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--log-every', type=int, default=200)
    p.add_argument('--ckpt-every', type=int, default=2000)
    p.add_argument('--device', default='auto', choices=['auto','cuda','cpu'])
    return p.parse_args()


if __name__ == '__main__':
    train(arguments())
