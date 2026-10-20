# Paper 2 — Large-Batch Training: Figure 3 Reproduction

Independent PyTorch reproduction of **Figure 3(a) F1, (c) C1, and (d) C2** from Keskar et al., *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima* (ICLR 2017), for CS 6362 Assignment 2.

**Phase 3 status:** completed independent implementation and measured interpolation experiments. No author implementation or external reproduction code was consulted during this phase. The curves are computed from trained networks, not hardcoded to match the paper.

## Scope and experiment

| Panel | Network | Dataset | Training examples | SB batch | LB batch |
| --- | --- | --- | ---: | ---: | ---: |
| F1 | Five-hidden-layer, 512-unit batch-normalized MLP | MNIST | 60,000 | 256 | 6,000 |
| C1 | Shallow AlexNet-inspired convolutional network | CIFAR-10 | 50,000 | 256 | 5,000 |
| C2 | Deep VGG-inspired convolutional network | CIFAR-10 | 50,000 | 256 | 5,000 |

Both regimes use **Adam**. The initial learning rate is **0.001**; the C1 SB/LB models were later resumed from epoch 50 with learning rate **0.0002** until epoch 120, retaining their optimizer state. Both regimes start from the **same initialization** within a panel (seed 42), an explicit reproduction choice.

For each network, evaluate the mean train/test cross-entropy and classification accuracy along

\[
\theta(\alpha)=(1-\alpha)\theta_{\mathrm{SB}}+\alpha\theta_{\mathrm{LB}},
\qquad \alpha\in[-1,2].
\]

The final experiment measures **61 equally spaced points**, including exact endpoints \(\alpha=0\) (SB) and \(\alpha=1\) (LB). No curve values are interpolated by the plotting notebook; each point is a model evaluation.

**BatchNorm choice:** reset and recompute running statistics at each interpolated weight configuration using the first **4,096 training examples**, with gradients disabled and dropout inactive. The full train/test splits are then evaluated. A separate diagnostic with linearly interpolated running statistics was unstable (including NaN losses during extrapolation); it is **not** used for the final figure. The 4,096-example recalibration policy is our implementation choice, not a claim about the paper's exact procedure.

## Repository layout

```text
paper2-large-batch/
├── README.md
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── interpolate.py
│   └── validate.py
├── notebooks/
│   └── reproduce_figure3.ipynb
└── results/
    └── final_61/
        ├── F1_interpolation.json
        ├── C1_interpolation.json
        ├── C2_interpolation.json
        └── selection_manifest.json
```

`results/final_61/` is the **minimal data needed to regenerate the plots**. It does **not** contain the six model checkpoints needed to rerun interpolation. Original and refined `endpoint.pt` / `last.pt` files and intermediate runs are archived separately on Google Drive under `MyDrive/ML-Paper-Reproductions/paper2/`. Do not delete that archive if you want to recompute Figure 3 from trained weights or resume training.

## Plot Figure 3 locally

1. Place the four final JSON files under `paper2-large-batch/results/final_61/`.
2. Open `paper2-large-batch/notebooks/reproduce_figure3.ipynb` in Jupyter or VS Code.
3. Run the cells in order.

The notebook follows the Paper 1 notebook conventions: it locates `results/final_61/` relative to the working directory, reads **measured data only**, and displays panels **F1, C1, C2** inline. It does not retrain models or generate new interpolation measurements. Blue curves represent cross-entropy (left axis), red curves accuracy (right axis); solid lines are training and dashed lines testing.

## Environment and datasets

From `paper2-large-batch/`:

```powershell
python -m pip install torch numpy matplotlib jupyter
```

For F1, obtain `data/mnist.npz` with arrays `x_train`, `y_train`, `x_test`, `y_test`:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
curl.exe -L --fail --retry 3 "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz" -o "data/mnist.npz"
```

For C1/C2, the default CIFAR-10 location is `../paper1-generalization/data/cifar-10-batches-py/`. Supply `--cifar-root` when using another location. MNIST and CIFAR-10 pixels are scaled to `[0,1]`; there is no cropping, per-image whitening, or data augmentation.

Check the setup:

```powershell
python src/data.py --panel F1
python src/data.py --panel C1
python src/validate.py --panel all --data
```

## Training and checkpointing

Example initial small-batch run:

```bash
python src/train.py --mode train --panel C1 --regime SB --device cuda --seed 42 --max-epochs 50 --output-root results/training
```

Replace `C1` and `SB` to train other panels and regimes. Batch sizes are assigned automatically. `last.pt` contains the model, Adam state, RNG states, and history for epoch-boundary resumption; `endpoint.pt` holds the model weights for inference; `metrics.json` contains recorded metrics.

**Example C1 fine-tuning continuation** (only after placing the original epoch-50 `last.pt` in the matching output folder):

```bash
python src/train.py --mode train --panel C1 --regime SB --device cuda --seed 42 --max-epochs 120 --resume-lr 0.0002 --patience 1000 --output-root results/refined
```

To reproduce the recorded continuation, first copy the baseline `C1_SB_seed42/` checkpoint directory to `results/refined/`; repeat independently for LB. `--max-epochs 120` is a **total** epoch limit, not 120 additional epochs. The learning-rate change is logged as a new stage.

**Limitation:** the default early stopping rule watches loss on a fixed 2,048-example prefix every five epochs and can stop before the entire training set is closely fitted. The refined C1 runs used an extended budget with plateau stopping effectively disabled. The final checkpoint selection is recorded in `selection_manifest.json`; it was based on training fit, not plot resemblance or test-set tuning.

## Rerun interpolation (requires archived checkpoints)

For a panel, restore both selected `endpoint.pt` files into a common output root:

```text
results/selected_endpoints/
├── C1_SB_seed42/endpoint.pt
└── C1_LB_seed42/endpoint.pt
```

Then run on a GPU-capable environment:

```bash
python src/interpolate.py --panel C1 --device cuda --output-root results/selected_endpoints --points 61 --bn-policy recalibrate --bn-examples 4096 --eval-batch 512
```

The script writes `C1_interpolation.json` to that root. Repeat for F1/C2 using their selected checkpoint pairs. It runs actual full-dataset forward evaluations; `--points 61` changes measurement resolution, not training.

## Phase 3 observations and limitations

Final **61-point recalibrated** endpoint measurements:

| Panel | SB train acc. | LB train acc. | SB test acc. | LB test acc. |
| --- | ---: | ---: | ---: | ---: |
| F1 | 99.823% | 100.000% | 98.15% | 98.38% |
| C1 | 99.988% | 99.794% | 70.79% | 66.44% |
| C2 | 99.718% | 99.610% | 86.50% | 81.10% |

The interpolation profiles qualitatively show different sensitivity around the SB/LB solutions, especially for C1/C2. The results **do not exactly reproduce** the published curves or test accuracies; F1 also does not show an SB test-accuracy advantage. Differences may arise from architecture assumptions, optimization and endpoint selection, shared initialization, and BN recalibration. These are observed deviations, **not data to adjust by hand**.

The paper reports five trials for its summary statistics; this project used one seed and one SB/LB pair per panel. The 1D plots describe the loss on **one parameter-space direction** and are not a global proof of sharpness.

## Phase boundary

This README documents the **independent Phase 3** work. Any later inspection of author code or changes made in Phase 4 should be clearly identified separately and should not overwrite the Phase 3 measurements or their provenance.
