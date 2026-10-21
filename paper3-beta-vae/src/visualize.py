"""Figure 7A/B for Higgins et al. (ICLR 2017), with comparable response scales.

The visualizations and diagnostics use *only trained inference checkpoints*.
All factor-conditioned means come from src.evaluate.calculate_factor_responses.

Changes relative to the original local plotting script:
  * common color scale across *all displayed* position heatmaps, both beta arms;
  * common Y scale per response type for all curves, both beta arms;
  * diverging map centered at zero; colorbar with actual latent-mean units;
  * numeric latent response amplitudes and per-dimension KL diagnostics in JSON;
  * factor-response variance decomposition (descriptive, NOT a disentanglement score);
  * same fixed reference factors and five traversal values for the two models.

No retraining, new β values, or pretrained model weights are involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from .data import DSprites, DEFAULT_DATA, index_from_factors
from .evaluate import (
    load_inference, calculate_factor_responses, reconstruction_diagnostics,
)

REFERENCE_FACTORS = (0, 1, 3, 10, 16, 16)  # [color, shape, scale, rot, x, y]
TRAVERSAL_VALUES = (3.0, 1.5, 0.0, -1.5, -3.0)
SHAPE_COLORS = ('red', 'green', 'blue')  # square, oval, heart


@torch.inference_mode()
def traverse(model, ds, device, latent_indices, reference_factors=REFERENCE_FACTORS):
    """Change exactly one coordinate of q(z|x)'s mean at a time.

    The other nine coordinates remain equal to encoder means of ONE fixed image.
    The reference image is disclosed; the authors do not specify their exact image.
    """
    idx = int(index_from_factors(np.asarray(reference_factors, dtype=np.int64)))
    x = ds.batch(np.asarray([idx], dtype=np.int64)).to(device=device, dtype=torch.float32)
    mu, _ = model.encode(x)
    images = []
    for value in TRAVERSAL_VALUES:
        z = mu.expand(len(latent_indices), -1).clone()
        for row, j in enumerate(latent_indices):
            z[row, j] = value
        pred = torch.sigmoid(model.decode_logits(z)).reshape(-1, 64, 64)
        images.append(pred.cpu().numpy())
    return np.stack(images)  # shape [5, number_of_columns, 64, 64]


def factor_response_diagnostics(response):
    """ANOVA-like main-effect signal from marginalized encoder response means.

    Since dSprites is a complete factorial dataset, variance of E[mu_j | v_k]
    indicates how strongly latent j responds to factor k. These are not independent
    importance estimates and should NOT be presented as a validated disentanglement
    score. They exclude the shape-main-effect (not separately aggregated here).
    """
    position = np.asarray(response['position_mean'], dtype=np.float64)  # [y,x,z]
    scale = np.asarray(response['scale_mean'], dtype=np.float64)        # [shape,6,z]
    rotation = np.asarray(response['rotation_mean'], dtype=np.float64)  # [shape,40,z]

    components = {
        'position_x': np.var(position.mean(axis=0), axis=0),
        'position_y': np.var(position.mean(axis=1), axis=0),
        'scale': np.var(scale, axis=1).mean(axis=0),
        'rotation': np.var(rotation, axis=1).mean(axis=0),
    }
    matrix = np.stack(list(components.values()), axis=0)
    totals = matrix.sum(axis=0)
    fractions = matrix / np.maximum(totals[None, :], 1e-12)
    spans = {
        'position': np.ptp(position, axis=(0, 1)),
        'scale': np.ptp(scale, axis=(0, 1)),
        'rotation': np.ptp(rotation, axis=(0, 1)),
    }
    return {
        'marginal_signal_variances': {k: v.tolist() for k, v in components.items()},
        'within_4_factor_signal_fractions': {
            k: fractions[i].tolist() for i, k in enumerate(components)
        },
        'raw_encoder_mean_spans': {k: v.tolist() for k, v in spans.items()},
        'note': (
            'Main-effect variance of conditional encoder means on the full factorial '
            'dataset. This is descriptive only; it excludes shape main effects, '
            'interactions, and statistical robustness. Not the paper disentanglement metric.'
        ),
    }


def data_for_panel(ckpt_path: Path, ds: DSprites, device, eval_batch_size: int, reference_factors=REFERENCE_FACTORS):
    model, meta = load_inference(ckpt_path, device)
    response = calculate_factor_responses(model, ds, device, eval_batch_size)
    order = [int(i) for i in response['sorted_latents']]
    selected = order[:5] + [order[-1]]  # five lowest posterior variances + one high
    generated = traverse(model, ds, device, selected, reference_factors)
    diagnostics = reconstruction_diagnostics(model, ds, device)
    return {
        'meta': meta, 'factors': response, 'selected': selected,
        'generated': generated, 'diagnostics': diagnostics,
        'factor_diagnostics': factor_response_diagnostics(response),
    }


def shared_limits(panels):
    """One set of absolute, non-robust limits, shared by both models if present.

    Only displayed dimensions enter the scale calculation. No per-column or
    per-latent normalization, no clipping, and no hardcoded latent amplitudes.
    """
    out = {}
    for key in ('position_mean', 'scale_mean', 'rotation_mean'):
        peak = max(
            float(np.max(np.abs(p['factors'][key][..., p['selected']])))
            for p in panels
        )
        out[key] = max(peak, 1e-6)
    return out


def draw_panel(panel, ax, label, limits):
    response = panel['factors']
    dims = panel['selected']
    generated = panel['generated']
    pvar = response['posterior_variance']
    kls = response['kl_per_latent']
    im = None

    for c, j in enumerate(dims):
        im = ax[0, c].imshow(
            response['position_mean'][:, :, j], origin='lower',
            cmap='jet',
            norm=TwoSlopeNorm(vmin=-limits['position_mean'], vcenter=0.0,
                              vmax=limits['position_mean']),
            interpolation='nearest',
        )
        active_label = ' (low KL)' if kls[j] < 0.05 else ''
        ax[0, c].set_title(f'z{j+1}{active_label}', fontsize=9)
        ax[0, c].set_xticks([])
        ax[0, c].set_yticks([])
        for spine in ax[0, c].spines.values():
            spine.set_visible(False)

        for r, key in ((1, 'scale_mean'), (2, 'rotation_mean')):
            target = ax[r, c]
            values = response[key][:, :, j]
            for s, color in enumerate(SHAPE_COLORS):
                target.plot(np.arange(values.shape[1]), values[s], color=color,
                            linewidth=1.1)
            target.set_ylim(-limits[key], limits[key])
            target.axhline(0.0, color='0.65', linewidth=0.4, zorder=0)
            target.set_xlim(0, values.shape[1] - 1)
            target.tick_params(bottom=False, left=False, labelbottom=False,
                               labelleft=False)
            target.grid(False)

        for r, _ in enumerate(TRAVERSAL_VALUES):
            target = ax[3 + r, c]
            target.imshow(generated[r, c], cmap='gray', vmin=0, vmax=1,
                          interpolation='nearest')
            target.set_xticks([])
            target.set_yticks([])

        # Means of q_phi(z_j | x)'s learned variance, NOT variance of mu_j.
        ax[7, c].set_xlabel(f'Var {pvar[j]:.3f}\nKL {kls[j]:.3f}', fontsize=7)

    for r, name in enumerate(('position', 'scale', 'rotation', '+3', '+1.5',
                              '0', '-1.5', '-3')):
        ax[r, 0].set_ylabel(name, fontsize=8, rotation=90, labelpad=6)
    ax[0, 0].text(0, 1.45, 'A: β-VAE (β=4)' if label == 'A' else 'B: VAE (β=1)',
                  transform=ax[0, 0].transAxes, weight='bold', fontsize=12)
    return im


def make_one(panel, label, dest: Path, limits):
    fig, axes = plt.subplots(
        8, 6, figsize=(11, 15),
        gridspec_kw={'height_ratios': [1.05, 1.05, 1.05, 1, 1, 1, 1, 1]},
    )
    image = draw_panel(panel, axes, label, limits)
    fig.subplots_adjust(wspace=.08, hspace=.18, left=.08, right=.93,
                        top=.95, bottom=.08)
    cb = fig.colorbar(image, ax=list(axes[0]), fraction=0.012, pad=.012)
    cb.set_label('Encoder mean μ (shared scale)', fontsize=8)
    fig.savefig(dest, dpi=160, bbox_inches='tight')
    plt.close(fig)


def make_joint(a, b, dest: Path, limits):
    fig, axes = plt.subplots(
        8, 12, figsize=(21, 15),
        gridspec_kw={'height_ratios': [1.05, 1.05, 1.05, 1, 1, 1, 1, 1]},
    )
    image = draw_panel(a, axes[:, :6], 'A', limits)
    draw_panel(b, axes[:, 6:], 'B', limits)
    for r in range(8):
        axes[r, 6].set_ylabel('')
    fig.subplots_adjust(wspace=.07, hspace=.18, left=.04, right=.94,
                        top=.95, bottom=.08)
    cb = fig.colorbar(image, ax=list(axes[0]), fraction=0.012, pad=.012)
    cb.set_label('Encoder mean μ (shared A/B scale)', fontsize=8)
    fig.savefig(dest, dpi=160, bbox_inches='tight')
    plt.close(fig)


def report(panel, limits, reference_factors=REFERENCE_FACTORS):
    response = panel['factors']
    meta = panel['meta']
    return {
        'beta': float(meta['beta']), 'seed': int(meta['seed']),
        'steps': int(meta['step']),
        'mean_posterior_var_all_10': [float(x) for x in response['posterior_variance']],
        'mean_kl_all_10': [float(x) for x in response['kl_per_latent']],
        'ordering_by_ascending_mean_posterior_variance_0_based':
            [int(i) for i in response['sorted_latents']],
        'displayed_latent_ids_1_based': [i + 1 for i in panel['selected']],
        'reconstruction': panel['diagnostics'],
        'factor_response_diagnostics': panel['factor_diagnostics'],
        'shared_plot_limits': limits,
        'reference_factor_classes': list(reference_factors),
        'traversal_values': list(TRAVERSAL_VALUES),
        'plot_note': 'Zero-centered absolute shared limits; no per-column color or Y autoscaling.',
    }


def run(args):
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    ds = DSprites(args.data, args.cache_dir)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    a = data_for_panel(args.output_root / 'beta4' / 'model.pt', ds, device,
                       args.eval_batch_size, args.reference_factors) if args.panel in ('A', 'both') else None
    b = data_for_panel(args.output_root / 'beta1' / 'model.pt', ds, device,
                       args.eval_batch_size, args.reference_factors) if args.panel in ('B', 'both') else None
    limits = shared_limits([p for p in (a, b) if p is not None])

    results = {}
    if a is not None:
        make_one(a, 'A', args.figures_dir / 'figure7_A.png', limits)
        results['A'] = report(a, limits, args.reference_factors)
        print('Figure 7A ready', json.dumps({
            'step': results['A']['steps'],
            'latents': results['A']['displayed_latent_ids_1_based'],
            'kl': [round(k, 4) for k in a['factors']['kl_per_latent']],
        }), flush=True)
    if b is not None:
        make_one(b, 'B', args.figures_dir / 'figure7_B.png', limits)
        results['B'] = report(b, limits, args.reference_factors)
        print('Figure 7B ready', json.dumps({
            'step': results['B']['steps'],
            'latents': results['B']['displayed_latent_ids_1_based'],
            'kl': [round(k, 4) for k in b['factors']['kl_per_latent']],
        }), flush=True)
    if a is not None and b is not None:
        make_joint(a, b, args.figures_dir / 'figure7_AB.png', limits)
    with (args.figures_dir / 'figure7_metrics.json').open('w') as f:
        json.dump(results, f, indent=2)
    print('Plot limits (same for BOTH beta arms):', json.dumps(limits))
    print('Output:', args.figures_dir, flush=True)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', choices=['A', 'B', 'both'], default='both')
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--output-root', type=Path,
                   default=Path('/content/paper3_runs' if Path('/content').exists()
                                else './results'))
    p.add_argument('--figures-dir', type=Path, default=Path('./figures'))
    p.add_argument('--eval-batch-size', type=int, default=2048)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--reference-factors', type=int, nargs=6, default=REFERENCE_FACTORS,
                   metavar=('COLOR','SHAPE','SCALE','ROT','X','Y'),
                   help='Unsupervised traversal reference image by dSprites factor classes')
    return p.parse_args()


if __name__ == '__main__':
    run(arguments())
