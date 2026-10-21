# CS 6362 — Paper 3: β-VAE, Figure 7A and 7B

Independent PyTorch implementation from the *paper only*, with no inspection or reuse of existing β-VAE implementations. This covers **Phase 1–3 code**; experiments are not claimed complete until real GPU training and comparison have actually been performed. **Do not inspect author code until AFTER preserving a Phase 3 commit.**

## Download the local dataset (manual; no SSL in Python)

1. Go to the [official Google DeepMind dSprites NPZ](https://github.com/google-deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz).
2. Click **Download raw file**. The official `.npz` is about **18.4 MB compressed**; its images expand to approximately **3.0 GB**.
3. Place it as `paper3-beta-vae/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`. **Do not rename or unzip it yourself.**
4. The `.gitignore` excludes datasets and checkpoints. For Colab, **also upload the same NPZ** to Google Drive at `MyDrive/ML-Paper-Reproductions/datasets/dsprites/` (the notebook copies it from Drive into fast Colab local storage). A dataset excluded from Git cannot be obtained by cloning your repo.
5. Dataset use is entirely offline; code never downloads files. In Colab the runtime's `data/` is populated by the setup notebook from Drive. A 3 GB extracted image cache is created on local disk on first load, then reused for training and evaluations. Ensure several GB of free local disk space.

Only use dataset labels/metadata for **plotting and verification**, never for VAE training. The genuine images have shape `(737280,64,64)` and uint8 values 0 or 1. Original factor names and dimensions: `(color=1, shape=3, scale=6, orientation=40, posX=32, posY=32)`. Mapping from factor classes to row index uses C-order flattening (posY fastest). The original dataset includes `latents_values`, `latents_classes`, `metadata`.

## Project layout

```text
paper3-beta-vae/
  data/                    # put the official NPZ here; never commit it
  src/data.py              # validate factor metadata; disk memmap cache; vectorized minibatches
  src/models.py            # architecture; analytic KL; Bernoulli reconstruction
  src/train.py             # train/resume each beta; minimal checkpoint + logs
  src/evaluate.py          # exact factor-conditioned encoder response (full dataset)
  src/visualize.py         # original Figure 7A/B response grids and decoder traversal
  src/validate.py          # math, shapes, gradients, resume roundtrip, optional benchmark
  notebooks/train_colab.ipynb
  notebooks/reproduce_figure7.ipynb
  results/ figures/        # ignored outputs (except .gitkeep)
```

## Reproduction specification — source of truth is Higgins et al., ICLR 2017

- **Figure 7A** is β-VAE **β=4**; **Figure 7B** is VAE **β=1**, both 2D Shapes.
- Appendix A.1, Table 1 specifies **flattened 4096-pixel input → encoder FC1200/FC1200 ReLU → 10 Gaussian latents → decoder FC1200/FC1200/FC1200/FC4096, Tanh, Bernoulli**; **Adagrad, learning rate 1e-2**.
- Prior `N(0,I)`. Loss per image: **summed binary cross-entropy across 4096 pixels + β × summed analytic Gaussian KL across 10 latents**. All batches average per example; logit outputs use numerically stable BCE-with-logits, decoder image visualization uses sigmoid. This scaling is essential (pixel-mean BCE would effectively multiply β by 4096).
- **Author's unspecified or ambiguous details**: we apply Tanh to all **decoder hidden** layers and no final Tanh (final layer returns logits of Bernoulli pixel probabilities); use default PyTorch `nn.Linear` initialization, Adagrad defaults, zero weight decay, no KL scheduling, no batch normalization; choose batch 256 and 12,000 optimizer updates (revisable based on benchmark). The paper does not prescribe batch size, seed, update count, exact initial weights, or an original public figure-generation script here. These are **our disclosed choices**, not invented original settings.
- Performance choice: enable TF32 matmul on NVIDIA CUDA GPUs for speed; this may introduce small floating-point differences relative to full FP32 (no autocast/FP16 is enabled).
- Controlled comparison: **one seed (2026) each**, same initial random weights for both β arms, identical minibatch permutation and training budget. β changes only the KL coefficient. One seed documents a particular run, **not robustness, statistical significance, or guaranteed disentanglement**.
- **Figure 7 columns are sorted by ascending mean inferred Gaussian posterior variance** (more active dimensions have smaller posterior variance). We display five most active latents and the most inactive as `z(inactive)`. No semantic latent IDs are hardcoded or cherry-picked. Display z labels are 1-based (as in paper). Latent assignments can differ from authors' particular random run.
- Figure's **position heatmap**: average mean encoder output by each of 32×32 x/y locations across all shapes/scales/orientations. **Scale plots**: mean encoder output by six scales within each of three shapes, averaged over orientations and positions. **Rotation plots**: response by 40 rotations, within each shape, averaged over scales and positions. Colors are red square, green oval, blue heart. `evaluate.py` performs *one full-data encoder pass* for all these aggregates; no ground truth is fed to training.
- **Five decoder images per latent**: choose one reproducible reference dSprites image `classes=[0,1,3,10,16,16]` (shape=oval, scale=3, orientation=10, x=y=16); encode it once to posterior mean; replace just the selected coordinate with `[+3,+1.5,0,-1.5,-3]` and decode, holding all nine other inferred means fixed. This is the caption's ±3 standard-deviation traversal around standard prior. Fixed reference image and five-step grid are disclosed choices where the caption is not more detailed.
- Paper's Fig. 7 position heatmaps and factor curves are **posterior encoder means**, not training-loss plots. Learned variance is `E_x[exp(logvar(x))]`, not `Var_x[mu(x)]`.

## Setup and local usage

Use Python 3.10+ with `torch`, `numpy`, `matplotlib`. Your repository-root requirements already contain torch and numpy; **add `matplotlib>=3.8` to the shared root requirements if it is not present**. Run commands from the **paper3-beta-vae parent**, e.g. from the repository root, so Python can resolve the folder with a hyphen, or run from `paper3-beta-vae/` with `python -m src.*`. Below assumes the latter:

```bash
cd paper3-beta-vae
python -m src.validate --paper-model
python -m src.validate --data data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
python -m src.validate --data data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz --benchmark --device cuda --max-steps 12000
python -m src.train --beta 1 --max-steps 12000 --batch-size 256 --output-root results
python -m src.train --beta 4 --max-steps 12000 --batch-size 256 --output-root results
python -m src.visualize --panel both --output-root results --figures-dir figures
```

**Important:** Benchmarks estimate throughput from observed training steps including actual local data reads. Do not assume 12,000 steps will be enough for faithful visual reproduction: inspect losses, posterior variances, and traversal grids and extend the budget using `--max-steps` **with the same other options** if appropriate. `--max-steps` is *total target steps*, not additional steps.

Output: `figures/figure7_A.png`, `figure7_B.png`, `figure7_AB.png`, `figure7_metrics.json`; `results/beta1/` and `beta4/` contain `latest.pt` resumable snapshots (including optimizer/RNG) and `model.pt` compact inference snapshots. If both trained models exist, visualization never retrains them. After the Colab run, download or copy final output/checkpoints into corresponding local folders for submission as appropriate; do not commit large weights.

## Google Colab

1. **Extract this ZIP into your private repository root**, creating `paper3-beta-vae/`, then push **only the source/notebooks** to the private GitHub repo. Do not make the repo public until assignment submission.
2. Download NPZ manually, upload to the described Drive folder. The .gitignore explicitly excludes this local 18 MB NPZ from GitHub.
3. Open `notebooks/train_colab.ipynb` in Colab and select a GPU (A100 or T4). Add `GITHUB_TOKEN` as a Colab Secret. It clones using a temporary `GIT_ASKPASS` script that reads the token from environment (no token in Git URL / remote). If already cloned, it will reuse that repo.
4. Run setup/Drive/data/smoke/benchmark; review *measured wall time* before running `β=1` then `β=4` **separately**. Small periodic backups go to `MyDrive/ML-Paper-Reproductions/paper3/`. Interrupted Colab session automatically resumes from backup `latest.pt` at the next batch of the same epoch. Inference checkpoint is also backed up.
5. `notebooks/reproduce_figure7.ipynb` uses model checkpoints in `/content/paper3_runs/{beta1,beta4}/model.pt`. If a new runtime, it copies them from Drive, then computes response plots from genuine model inference. Evaluation of all 737k images may take time, so it runs once per model.
6. Training and plotting notebooks do **not** silently start extra seeds/ablation grids or call any download APIs to fetch the dataset.

## Research notes & decisions for report

- `L_β = E[log p(x|z)] - β KL(q(z|x)||p(z))`: β>1 applies stronger KL pressure, sometimes improving independence but risking inactive units and lower reconstruction quality. β<1 relieves this pressure, typically improving fit but allowing more entanglement. β=1 standard ELBO. **Neither β choice guarantees a successful visual result.**
- Record both measured reconstruction cost and KL, posterior variances/KL by latent, training steps and wall time, generated Figure 7A/B, full dataset factor-response figures, and seed=2026. `figure7_metrics.json` documents the empirically selected latent order and deterministic sample quality. Reconstruction evaluation is on a **subset of the training distribution**, not a held-out test dataset.
- A successful replication means β=4 has visibly more selective latent curves and single-coordinate decoder transformations than β=1; don't conflate color/shape priors, selection bias, or pretty traversal images with quantitative proof. If it fails, inspect optimizer, loss scale, posterior KL, inactive units, and convergence before changing architecture/regularization.
- Fig. 7 shows one illustrative result; your project trains **one independent seed per β**. Statistical claims would require more seeds and different selection policies. Keep this as an explicit limitation.
- **Phase 2 checks built into `validate.py`:** objective sign/scaling, β difference, KL near zero when q=p, forward/gradients finite, checkpoint roundtrip, correct factor index mapping, archive verification. No corrections are fabricated; note fixes made after actual debugging in your report.
- **Phase 3**: train, inspect outcomes, diagnose deviations with evidence, then commit source and a summary of real outputs and record Git SHA **before Phase 4**. **Phase 4 is not performed here**. Only then compare with any author implementation if you want, documenting differences.

### Scientific references

Higgins et al., *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*, ICLR 2017, Sections 2, 4.2, Figure 7 and Appendix A.1 Table 1. Assignment 2, CS 6362 (Fall 2026), Paper 3 instructions and reproduction-phase rubric. dSprites dataset by Matthey, Higgins, Hassabis and Lerchner, 2017: https://github.com/google-deepmind/dsprites-dataset .
