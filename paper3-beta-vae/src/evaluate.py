"""Exact factor-conditioned encoder means for all 737280 official dSprites.

Single inference-only pass aggregates all examples into 32x32 position maps,
shape-specific scale/rotation curves, and global posterior variance per unit.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from .data import DSprites, DEFAULT_DATA, FACTOR_SIZES, N_IMAGES
from .models import BetaVAE, ModelConfig


def load_inference(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = ModelConfig(**ckpt['model_config'])
    if config.latent_dim != 10 or config.input_dim != 4096:
        raise ValueError('Figure 7 requires 10 latent units and 64x64 pixels')
    model = BetaVAE(config).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model, ckpt


@torch.inference_mode()
def calculate_factor_responses(model: BetaVAE, ds: DSprites, device: torch.device,
                               batch_size: int = 2048) -> dict:
    """Means are truly averaged over all other factors, matching paper's caption."""
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    z = model.config.latent_dim
    # The group counts are known from full Cartesian product.
    # row 0 heatmap index is x*32+y, reshaped to [y,x] for conventional image axes.
    position = torch.zeros(1024, z, device=device, dtype=torch.float64)
    scale = torch.zeros(3*6, z, device=device, dtype=torch.float64)
    rotation = torch.zeros(3*40, z, device=device, dtype=torch.float64)
    sum_var = torch.zeros(z, device=device, dtype=torch.float64)
    sum_kl = torch.zeros(z, device=device, dtype=torch.float64)
    for start in range(0, N_IMAGES, batch_size):
        end = min(start + batch_size, N_IMAGES)
        ids = np.arange(start, end, dtype=np.int64)
        # Ordered axes [color, shape, scale, orientation, posX, posY].
        shape = (ids // (6*40*32*32)) % 3
        size = (ids // (40*32*32)) % 6
        angle = (ids // (32*32)) % 40
        pos_x = (ids // 32) % 32
        pos_y = ids % 32
        x = ds.batch(ids).to(device=device, dtype=torch.float32)
        mu, lv = model.encode(x)
        m = mu.to(dtype=torch.float64)
        position.index_add_(0, torch.as_tensor(pos_x*32+pos_y, device=device), m)
        scale.index_add_(0, torch.as_tensor(shape*6+size, device=device), m)
        rotation.index_add_(0, torch.as_tensor(shape*40+angle, device=device), m)
        sum_var += lv.exp().double().sum(dim=0)
        sum_kl += (-0.5*(1 + lv - mu.square() - lv.exp())).double().sum(dim=0)
    out = {
        'posterior_variance': (sum_var / N_IMAGES).cpu().numpy(),
        'kl_per_latent': (sum_kl / N_IMAGES).cpu().numpy(),
        # image reshape as [Y, X, z] for imshow origin=lower
        'position_mean': (position / (3*6*40)).cpu().numpy().reshape(32,32,z).transpose(1,0,2),
        'scale_mean': (scale / (40*32*32)).cpu().numpy().reshape(3,6,z),
        'rotation_mean': (rotation / (6*32*32)).cpu().numpy().reshape(3,40,z),
    }
    out['sorted_latents'] = np.argsort(out['posterior_variance'])
    return out


@torch.inference_mode()
def reconstruction_diagnostics(model: BetaVAE, ds: DSprites, device: torch.device,
                               sample_size: int = 2048, batch_size: int = 256, seed: int = 777):
    """Use deterministic encoder means; descriptive in-distribution sample, NOT held-out test."""
    from torch.nn import functional as F
    rng = np.random.default_rng(seed)
    ids = rng.choice(N_IMAGES, min(sample_size, N_IMAGES), replace=False)
    tot_bce = 0.0
    tot_kl = 0.0
    tot_binary_accuracy = 0.0
    for start in range(0, len(ids), batch_size):
        x = ds.batch(ids[start:start+batch_size]).to(device=device, dtype=torch.float32)
        logits, mu, lv = model(x, deterministic=True)
        tot_bce += float(F.binary_cross_entropy_with_logits(logits, x, reduction='sum').item())
        tot_kl += float((-0.5*(1+lv-mu.square()-lv.exp())).sum().item())
        tot_binary_accuracy += float(((logits >= 0) == (x >= 0.5)).sum().item())
    n = len(ids)
    return {'n_images': n, 'bce_per_image': tot_bce/n,
            'kl_per_image': tot_kl/n, 'pixel_accuracy': tot_binary_accuracy/(n*4096),
            'note': 'Reconstruction at deterministic posterior mean; descriptive subset of training distribution.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--device', default='auto', choices=('auto','cuda','cpu'))
    a=p.parse_args()
    device = torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device)
    ds=DSprites(a.data,a.cache_dir)
    model, ckpt=load_inference(a.checkpoint,device)
    response=calculate_factor_responses(model,ds,device,a.batch_size)
    print('beta:',ckpt['beta'],'seed:',ckpt['seed'],'steps:',ckpt['step'])
    print('posterior variance:',np.round(response['posterior_variance'],3))
    print('latent ordering (0-based):',response['sorted_latents'])
    print('reconstruction:',reconstruction_diagnostics(model,ds,device))


if __name__ == '__main__':
    main()
