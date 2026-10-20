"""Phase 4 trainer, CUDA AMP variant for limited-VRAM GPUs (e.g. Tesla T4).

Preserves the original effective batch sizes (SB=256, LB=5000) and
BatchNorm's full-batch behavior. CUDA autocast fp16 plus GradScaler changes
numerical precision relative to Phase 3; this is a documented Phase 4 choice.
Checkpoints store optimizer/scaler/RNG states and can sync to Drive.
"""
import argparse
import json
import os
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import load_data, loader
from train import eval_model, set_seed
from phase4_models import VERSION, create


def atomic_save(obj, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, dest)


def backup(folder, backup_root, names):
    if not backup_root:
        return
    target = Path(backup_root) / folder.name
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = folder / name
        if not src.is_file():
            continue
        tmp = target / (name + '.partial')
        shutil.copy2(src, tmp)
        os.replace(tmp, target / name)
    print('Drive sync:', target, flush=True)


def make_precision(device, precision):
    if precision == 'auto':
        precision = 'fp16' if device == 'cuda' else 'fp32'
    if precision == 'fp16' and device != 'cuda':
        raise ValueError('FP16 autocast requires a CUDA device')
    if precision not in ('fp16', 'fp32'):
        raise ValueError(f'Unsupported precision: {precision}')
    scaler = torch.amp.GradScaler('cuda', enabled=(precision == 'fp16'))
    return precision, scaler


def train_step(model, optimizer, scaler, x, y, precision):
    optimizer.zero_grad(set_to_none=True)
    context = torch.autocast(device_type='cuda', dtype=torch.float16) if precision == 'fp16' else nullcontext()
    with context:
        logits = model(x)
        # Compute cross-entropy in FP32 even when logits were produced in FP16.
        loss = nn.functional.cross_entropy(logits.float(), y)
    if precision == 'fp16':
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return loss.detach()


def run(a):
    if a.panel not in ('C1', 'C2'):
        raise ValueError('F1 is unchanged in Phase 4; reuse Phase 3 results.')
    if a.max_epochs < 1 or a.checkpoint_every < 1:
        raise ValueError('max-epochs and checkpoint-every must be >= 1')
    if a.bench_steps < 1:
        raise ValueError('bench-steps must be >= 1')

    device = 'cuda' if a.device == 'auto' and torch.cuda.is_available() else ('cpu' if a.device == 'auto' else a.device)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    precision, scaler = make_precision(device, a.precision)
    set_seed(a.seed)
    ds, test = load_data(a.panel, a.cifar_root, a.mnist_file)
    batch = 256 if a.regime == 'SB' else len(ds) // 10
    config = dict(panel=a.panel, regime=a.regime, seed=a.seed, batch=batch,
                  initial_lr=a.lr, architecture_version=VERSION,
                  data='raw/255; no augmentation', optimizer='Adam',
                  precision=precision)
    root = Path(a.output_root) / f'{a.panel}_{a.regime}_seed{a.seed}'
    model = create(a.panel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    print('CONFIG', config, 'device', device,
          'parameters', sum(p.numel() for p in model.parameters()), flush=True)
    if device == 'cuda':
        print('GPU', torch.cuda.get_device_name(0), 'total_GiB',
              round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), flush=True)
        torch.cuda.reset_peak_memory_stats()

    if a.mode == 'benchmark':
        it = iter(loader(ds, batch, shuffle=True, device=device, seed=a.seed))
        count = 0
        t0 = time.perf_counter()
        model.train()
        for _ in range(a.bench_steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader(ds, batch, shuffle=True, device=device, seed=a.seed))
                x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            train_step(model, optimizer, scaler, x, y, precision)
            count += len(y)
        if device == 'cuda':
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        print(json.dumps(dict(samples=count, seconds=secs,
                              precision=precision, batch=batch,
                              estimated_epoch_seconds=len(ds)*secs/count,
                              peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None), indent=2))
        return

    if a.mode == 'smoke':
        x, y = next(iter(loader(ds, 16, device=device)))
        model.train()
        loss = train_step(model, optimizer, scaler,
                          x.to(device), y.to(device), precision)
        print('SMOKE OK', float(loss.cpu()), 'precision', precision)
        return

    root.mkdir(parents=True, exist_ok=True)
    ck = root / 'last.pt'
    start_epoch = 0
    history = []
    lr_stages = []
    if ck.is_file():
        saved = torch.load(ck, map_location='cpu', weights_only=False)
        if saved['config'] != config:
            raise RuntimeError('Checkpoint configuration mismatch (including precision); '
                               'use a new output root or restore a matching checkpoint.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if precision == 'fp16':
            if 'scaler' not in saved:
                raise RuntimeError('Cannot resume AMP from checkpoint without scaler state.')
            scaler.load_state_dict(saved['scaler'])
        start_epoch = saved['epoch']
        history = saved['history']
        lr_stages = saved.get('lr_stages', [])
        torch.set_rng_state(saved['torch_rng'])
        if torch.cuda.is_available() and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        print('RESUMED at epoch', start_epoch, flush=True)
    elif a.resume_lr is not None:
        raise RuntimeError('--resume-lr needs an existing checkpoint.')

    if a.resume_lr is not None:
        previous = float(optimizer.param_groups[0]['lr'])
        if abs(previous - a.resume_lr) > 1e-12:
            for group in optimizer.param_groups:
                group['lr'] = a.resume_lr
            lr_stages.append(dict(after_epoch=start_epoch, from_lr=previous, to_lr=a.resume_lr))
            print('Learning-rate stage', previous, '->', a.resume_lr, flush=True)
    if start_epoch >= a.max_epochs:
        print('Already at or beyond requested epoch budget:', start_epoch)
        backup(root, a.backup_root, ['last.pt', 'endpoint.pt', 'metrics.json'])
        return

    t0 = time.monotonic()
    for epoch in range(start_epoch + 1, a.max_epochs + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        n = 0
        for x, y in loader(ds, batch, shuffle=True, device=device, seed=a.seed+epoch):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = train_step(model, optimizer, scaler, x, y, precision)
            total_loss += loss * len(y)
            n += len(y)
        rec = dict(epoch=epoch, training_batch_loss=float((total_loss/n).item()),
                   lr=float(optimizer.param_groups[0]['lr']))
        if epoch % a.eval_every == 0 or epoch == a.max_epochs:
            # Evaluation intentionally remains full FP32, as in Phase 3.
            rec['train_probe'] = eval_model(model, ds, a.eval_batch, device, 2048)
            rec['test_probe'] = eval_model(model, test, a.eval_batch, device, 2048)
        history.append(rec)
        print(f'{a.panel}_{a.regime} epoch={epoch} train_step_CE={rec["training_batch_loss"]:.5f}' +
              (f' probe_train_acc={rec["train_probe"]["accuracy"]*100:.2f}% probe_test_acc={rec["test_probe"]["accuracy"]*100:.2f}%'
               if 'train_probe' in rec else ''), flush=True)
        if epoch % a.checkpoint_every == 0 or epoch == a.max_epochs:
            state = dict(config=config, epoch=epoch, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         history=history, lr_stages=lr_stages,
                         torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None,
                         numpy_rng=np.random.get_state(), python_rng=random.getstate())
            atomic_save(state, ck)
            backup(root, a.backup_root, ['last.pt'])

    final = dict(config=config, epoch=a.max_epochs, stop_reason='budget_exhausted',
                 train=eval_model(model, ds, a.eval_batch, device),
                 test=eval_model(model, test, a.eval_batch, device),
                 lr_stages=lr_stages, history=history, elapsed_s=time.monotonic()-t0)
    (root / 'metrics.json').write_text(json.dumps(final, indent=2))
    atomic_save(dict(config=config,
                     model={k:v.cpu() for k,v in model.state_dict().items()},
                     final=final), root / 'endpoint.pt')
    backup(root, a.backup_root, ['last.pt', 'metrics.json', 'endpoint.pt'])
    print('FINAL', json.dumps({k:v for k,v in final.items() if k not in ('history',)}, indent=2), flush=True)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', choices=['C1','C2'], required=True)
    p.add_argument('--regime', choices=['SB','LB'], required=True)
    p.add_argument('--mode', choices=['benchmark','smoke','train'], default='benchmark')
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--resume-lr', type=float)
    p.add_argument('--max-epochs', type=int, default=50)
    p.add_argument('--checkpoint-every', type=int, default=5)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--eval-batch', type=int, default=512)
    p.add_argument('--bench-steps', type=int, default=8)
    p.add_argument('--output-root', default='results/phase4')
    p.add_argument('--backup-root')
    p.add_argument('--cifar-root')
    p.add_argument('--mnist-file')
    p.add_argument('--precision', choices=['auto','fp16','fp32'], default='auto',
                   help='auto uses CUDA fp16 AMP and CPU fp32; use fp32 to compare baseline')
    return p.parse_args()

if __name__ == '__main__':
    run(parse())
