# Advanced Machine Learning — Paper Reproductions

Independent PyTorch reproductions of three ICLR 2017 papers for **CS 6362: Advanced Machine Learning, Fall 2026 — Assignment 2**.

The assignment examines whether an LLM-assisted implementation can reproduce published results through genuine experiments rather than plots constructed to resemble the originals. Each project includes model and training code, evaluation/visualization notebooks, recorded results, and documentation of the decisions and failures encountered during reproduction.

## Papers and results

| Paper | Publication | Required reproduction | Outcome |
| --- | --- | --- | --- |
| [Paper 1 — Generalization](paper1-generalization/) | Zhang et al., *Understanding Deep Learning Requires Rethinking Generalization* (2017) | Figure 1(a–c) | Qualitative memorization/generalization trends reproduced; MLP convergence-time discrepancies documented. |
| [Paper 2 — Large-batch training](paper2-large-batch/) | Keskar et al., *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima* (2017) | Figure 3: F1, C1, C2 | Measured interpolation curves and small-batch advantages for C1/C2; F1 and some curve details differ. |
| [Paper 3 — β-VAE](paper3-beta-vae/) | Higgins et al., *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework* (2017) | Figure 7A/B | Genuine latent responses and decoder traversals; β=4 learns some factor-specific coordinates but does not cleanly isolate scale. |

These are **documented reproduction attempts, not claims of exact numerical replication**. Experimental conditions, caveats, and comparisons with the published figures are described in the respective paper directories and reports.

## Repository layout

```text
advanced-ml-paper-reproductions/
├── paper1-generalization/   # CIFAR-10 randomization experiments
├── paper2-large-batch/      # MNIST/CIFAR-10 SB versus LB interpolation
├── paper3-beta-vae/         # dSprites β=1 and β=4 models and traversals
├── requirements.txt        # Shared Python environment
└── README.md
```

Each subproject maintains its own `src/`, `notebooks/`, and experiment results. See the per-paper READMEs for exact paths, dataset setup, commands, and model selections:

- [Paper 1 instructions](paper1-generalization/README.md)
- [Paper 2 instructions](paper2-large-batch/README.md) and [Phase 4 notes](paper2-large-batch/README_PHASE4.md)
- [Paper 3 instructions](paper3-beta-vae/README.md)

## Getting started

Use Python and Jupyter in a virtual environment. The repository's `requirements.txt` is a pinned snapshot of the original working environment; GPU-compatible PyTorch installation may depend on the host and CUDA runtime.

```bash
python -m venv .venv
# Activate .venv using your shell's standard command
python -m pip install -r requirements.txt
```

Follow the relevant paper's README to obtain its dataset and execute its training or evaluation scripts. Experiments were primarily trained with **PyTorch on Google Colab GPUs**. Plotting notebooks generally read recorded measurements or load trained checkpoints; they should not be mistaken for complete training workflows by themselves.

| Paper | Main figure notebook | Input required |
| --- | --- | --- |
| 1 | `paper1-generalization/notebooks/reproduce_figure1.ipynb` | Recorded Figure 1 training logs |
| 2 | `paper2-large-batch/notebooks/reproduce_figure3.ipynb` | Measured Phase 3 interpolation JSONs |
| 3 | `paper3-beta-vae/notebooks/reproduce_figure7.ipynb` | Official dSprites dataset and trained seed-42 checkpoints |

Paper 1 and Paper 2 also include separate later-analysis notebooks. Paper 2's Phase 4 figure notebook uses Phase 3 F1 results and author-informed Phase 4 C1/C2 results; Paper 3's Phase 4 notebook is diagnostic rather than a replacement for the original Phase 3 submission.

### Large files and reproducibility

Some datasets and recorded results are present in the repository, but **do not assume every large training artifact is tracked directly by Git**. Paper 2's selected model/optimizer checkpoints are archived separately in Google Drive; its committed interpolation JSONs support plotting but not retraining-free recomputation of interpolation from weights. Paper 3 inference and resume checkpoints are tracked as **Git LFS objects**; use `git lfs pull` after cloning, and supply the official dSprites dataset as directed in that subproject's README. For a complete rerun, follow each paper's data and checkpoint instructions.

## Development protocol

The work followed the assignment's four-phase procedure:

1. **Phase 1:** Read the paper and draft an independent implementation through the LLM chat interface.
2. **Phase 2:** Inspect and validate the generated implementation, identifying and fixing errors.
3. **Phase 3:** Run and diagnose the independent reproduction, then preserve the implementation in Git **before examining existing implementations**.
4. **Phase 4:** Examine available author-released code, compare it with the saved Phase 3 version, and document any follow-up changes and experiments separately.

The Phase 3 and Phase 4 boundaries and experiment-specific corrections are documented inside each paper directory. The recorded plots are computed from measured training/evaluation outputs or trained models; visualization code does not embed manually invented outcome curves.

## Scope and limitations

The assignment permits carefully explained mismatches. Accordingly, the reports distinguish paper-prescribed settings from implementation choices, independent Phase 3 results from Phase 4 diagnostics, and single-seed illustrations from statistical replication. The reproduction notebooks and reported metrics should be interpreted together with their methodological caveats.

## References

- Zhang et al. (2017), [*Understanding Deep Learning Requires Rethinking Generalization*](https://openreview.net/forum?id=Sy8gdB9xx).
- Keskar et al. (2017), [*On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*](https://openreview.net/forum?id=H1oyRlYgg).
- Higgins et al. (2017), [*β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*](https://openreview.net/forum?id=Sy2fzU9gl).

This repository is an educational reproduction project. The original papers and any referenced author repositories remain the work of their respective authors.
