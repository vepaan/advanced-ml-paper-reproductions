# Paper 3 — β-VAE: Reproducing Figure 7A and 7B

**Course:** CS 6362 Advanced Machine Learning, Assignment 2  
**Paper:** Higgins et al. (2017), *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*  
**Scope:** Independent implementation and Phase 3 reproduction of Figure 7A (β = 4) and Figure 7B (β = 1), using the 2D Shapes/dSprites dataset.

This project was developed from the paper without consulting author or third-party reproduction code during Phases 1–3. The figures are produced from learned PyTorch checkpoint weights, not handcrafted latent responses, hardcoded image arrays, or reproduced values copied from the published figures. The results constitute a **partial qualitative reproduction**: the reconstruction–information tradeoff and several factor-specific latents are present, but all of the paper's disentanglement patterns were not recovered.

## Project files and reproducibility

```text
paper3-beta-vae/
├── data/
│   └── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz  # official data, supplied locally
├── src/
│   ├── data.py        # official dataset checks, indexing, disk-backed image cache
│   ├── models.py      # encoder/decoder and β-VAE objective
│   ├── train.py       # unsupervised training, checkpointing, deterministic resume
│   ├── evaluate.py    # dataset-wide encoder responses and reconstruction diagnostics
│   ├── visualize.py   # Figure 7A/B plots and traversals
│   └── validate.py    # mathematical checks and model/data smoke tests
├── results/
│   └── seed42/
│       ├── beta1/
│       │   ├── model.pt  # inference checkpoint
│       │   └── latest.pt # resumable training checkpoint (optional for plotting)
│       └── beta4/
│           ├── model.pt
│           └── latest.pt
└── notebooks/          # reproduction and experiment notebooks are organized separately
```

**Checkpoint handling:** The final seed-42 `.pt` files are intended to be stored with **Git LFS**, not ordinary Git blobs. Clone with Git LFS installed and pull LFS objects to obtain the checkpoint bytes. For figure generation, only `beta1/model.pt` and `beta4/model.pt` are required. The larger `latest.pt` files additionally hold optimizer/RNG state to resume training. The official `.npz` dataset must be supplied separately if it is not present in a checkout. Do not accidentally commit the extracted multi-gigabyte cache.

## Data and methods

Download the official dSprites archive from the [Google DeepMind dSprites dataset repository](https://github.com/google-deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz) and place the unchanged file in `data/` as shown above. The image tensor has **737,280** binary images of size **64 × 64**. Factor sizes, in dataset order, are **(1 color, 3 shapes, 6 scales, 40 orientations, 32 X positions, 32 Y positions)**. `src/data.py` checks the archive's factor metadata and index mapping, and extracts the image array into a reusable disk-backed `.npy` cache.

**Unsupervised training:** The training loop uses only image pixels. Ground-truth factor metadata is used to verify the dataset and to compute diagnostic plots *after* training, never as targets or labels in gradient updates.

### Architecture and objective

Following Appendix A.1, Table 1:

| Component | Configuration |
|---|---|
| Input | Flattened 64 × 64 binary image (4,096 values) |
| Encoder | FC 1200 → ReLU → FC 1200 → ReLU |
| Posterior | 10 Gaussian latent variables, learned mean and log variance |
| Prior | Standard normal N(0, I) |
| Decoder | FC 1200 → Tanh → FC 1200 → Tanh → FC 1200 → Tanh → FC 4096, Bernoulli logits |
| Optimizer | Adagrad, learning rate 0.01 |
| Figure 7A / 7B | β = 4 / β = 1 |

The minimized objective is the **negative β-ELBO**:

\[
\mathcal L_\beta = \frac{1}{B}\sum_{i=1}^{B}\left[
\operatorname{BCE}_{\rm sum}(x_i,\hat x_i) +
\beta\,D_{\rm KL}(q_\phi(z\mid x_i)\parallel\mathcal N(0,I))
\right].
\]

Reconstruction BCE is **summed over 4,096 pixels per example** before averaging over the batch; KL is summed over the 10 latent coordinates. This preserves the intended weighting of β. The implementation uses reparameterized Gaussian samples in training and numerically stable BCE-with-logits.

**Choices not established as exact author settings:** batch size 256, 30,000 total optimizer steps for the final models, seeds, default PyTorch linear-layer initialization and Adagrad defaults, no KL warmup, and a deterministic reference image for decoder traversals. The Bernoulli output is represented with logits rather than probabilities in the loss, and Tanh is applied to decoder *hidden* layers. CUDA training enables TF32 matmul; this may cause small numerical differences. These are disclosed implementation decisions, not claimed settings from the original experiment.

## Phase 3 experiment record

The initial independent comparison trained both β values with **seed 2026**. After reviewing latent-factor responses, β = 4 was extended to 30,000 steps and two additional β = 4 initializations were tested. A final **matched-seed** β = 1 model was then trained for seed 42 so that Figure 7A and 7B could use the same initialization and data-shuffling policy.

| Experiment | β | Seed | Updates | Purpose |
|---|---:|---:|---:|---|
| Initial baseline | 1 | 2026 | 30,000 | Original VAE baseline |
| Initial β-VAE | 4 | 2026 | 30,000 | Independent β-VAE run |
| Seed ablation | 4 | 42 | 30,000 | Alternate initialization; final illustrated β-VAE |
| Seed ablation | 4 | 123 | 30,000 | Third β-VAE initialization |
| Matched comparison | 1 | 42 | 30,000 | Final Figure 7B matched to β = 4, seed 42 |

All runs used the same 10-dimensional model, Adagrad at 0.01, batch size 256, official dataset, and fixed update count in the final comparison. Seed 42 was **chosen after examining multiple β = 4 outcomes**, so the final illustrated pair is a controlled comparison *conditional on that seed*, not an unbiased estimate of typical performance over random initializations. Retain the other runs' logs and checkpoints in the experiment archive if needed for the report.

### Observed outcomes

For the final **seed-42, 30,000-step** models, the recorded deterministic-posterior-mean reconstruction diagnostics were approximately:

| Diagnostic | β = 4 (Figure 7A) | β = 1 (Figure 7B) |
|---|---:|---:|
| BCE per image | 86.06 | 61.53 |
| Total mean KL | 9.58 | 17.37 |
| Latent dimensions with KL > 0.1 | 4 | 6 |

The β = 4 run learned clear horizontal-position, vertical-position, and rotation responses but **did not isolate scale cleanly**; the original figure displays stronger and more complete separation. The β = 1 run achieved better reconstruction and represented scale more distinctly in at least one latent, while several other coordinates mixed factors. Across the three β = 4 seeds, factor allocation changed materially even when reconstruction and KL magnitudes were similar. Thus the reproduction supports the compression/reconstruction tradeoff but **does not establish a universal disentanglement advantage for β = 4**.

These comparisons come from a deterministic image sample and the full-dataset encoder response statistics. The custom *factor-response signal fractions* measure variance of conditional latent means for X, Y, scale, and rotation; they **exclude shape main effects and interactions** and are **not the paper's formal classification-based disentanglement score**. Reconstruction evaluation uses in-distribution examples, not held-out test images. No formal multi-seed significance claim is made.

## Reproduce the final Figure 7A and Figure 7B locally

Use Python 3.10+ with `torch`, `numpy`, and `matplotlib`. Run all commands from **`paper3-beta-vae/`**; a GPU is optional for evaluation. First validate the code and dataset:

```powershell
python -m src.validate --paper-model
python -m src.validate --data data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

Verify that the two *inference* checkpoint files exist at `results/seed42/beta4/model.pt` and `results/seed42/beta1/model.pt`. To regenerate the full paper-style panels and metrics using the CLI:

```powershell
python -m src.visualize --panel both --data data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz --output-root results/seed42 --figures-dir outputs/figure7 --device auto --eval-batch-size 512
```

This command **creates** `outputs/figure7/figure7_A.png`, `figure7_B.png`, `figure7_AB.png`, and `figure7_metrics.json`. The plotting notebook can instead render the figures **inline, without creating a figures directory**. CLI output directories are optional generated artifacts, not required repository inputs. On CPU, evaluating the entire dataset separately through both encoders may take several minutes or longer.

To retrain either final model from scratch rather than use checkpoints, use a **new empty output directory** to avoid automatically resuming existing checkpoints:

```powershell
python -m src.train --beta 4 --seed 42 --max-steps 30000 --batch-size 256 --learning-rate 0.01 --data data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz --output-root results/retrain_seed42
python -m src.train --beta 1 --seed 42 --max-steps 30000 --batch-size 256 --learning-rate 0.01 --data data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz --output-root results/retrain_seed42
```

`--max-steps` denotes the **total** target step count, not additional steps. The trainer preserves model/optimizer/RNG state in `latest.pt` for exact continuation under matching settings and writes `model.pt` for inference. Training is non-deterministic across hardware/software environments in general, even with the same seed; numerical exactness beyond a fixed environment is not guaranteed.

## How Figure 7 is calculated

- **Position row:** each latent's encoder mean, averaged by the 32 × 32 object position grid across shapes, scales, and orientations.
- **Scale and rotation rows:** each latent's encoder mean as a function of one factor, with three colored traces for square (red), oval (green), and heart (blue); other relevant factors are averaged out.
- **Latent columns:** five smallest average inferred posterior variances plus one high-variance, near-inactive comparison. Selection is statistical; no handpicked semantic latent indices are passed to the plotter.
- **Decoder traversals:** the dataset reference image has factor classes `[0,1,3,10,16,16]`. The selected coordinate is set successively to `[+3,+1.5,0,-1.5,-3]`, while the other nine coordinates remain fixed to their inferred posterior means. Each modified vector is decoded through the learned network.
- **Comparable plotting scales:** encoder response scales are shared across both β models, and heatmap colors indicate the actual latent mean. No per-latent color autoscaling is used to exaggerate inactive coordinates.

As the paper's figure caption specifies, these panels show **encoder means and decoded traversals** rather than optimizer loss curves. The exact reference image, plotting aesthetics, and training run need not coincide with the paper's illustrative example. Latent labels can legitimately permute between independently trained runs.

## Phase boundaries and limitations

**Phases 1–3:** Independent paper interpretation, implementation, validation, actual training, and documented comparison. The Phase 3 Git commit must be preserved and its SHA recorded before inspecting an author implementation. No author or third-party reproduction code was used to create this implementation. Published qualitative patterns were not imposed on the generated curves or decoder outputs.

**Phase 4 (not included in this baseline):** Only after explicit approval and preservation of the Phase 3 snapshot, compare with released author code, document differences, and label any subsequent changes as Phase 4 work. Do not retroactively present Phase 4-informed changes as independent Phase 3 design.

### References

- Higgins et al. (2017), *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*, ICLR 2017, Sections 2–4, Figure 7, Appendix A.1/Table 1.
- [Google DeepMind dSprites dataset](https://github.com/google-deepmind/dsprites-dataset).
- CS 6362 Advanced Machine Learning, Fall 2026, Assignment 2.
