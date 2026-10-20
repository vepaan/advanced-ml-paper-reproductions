# Paper 2 / Figure 3 — independent, compute-conscious MVP

**Place the six Python files in `paper2-large-batch/src/` and retain this README under `paper2-large-batch/`.** No pretrained models or architecture reference code used. All figures must derive from measured data.

## Environment

Run from the **`paper2-large-batch`** directory (same Python environment as Paper 1 is sufficient):

```powershell
python -m pip install torch numpy
```

### Manual MNIST download (avoids Python HTTPS / SSL)

From `paper2-large-batch` in **PowerShell**:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
curl.exe -L --fail --retry 3 "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz" -o "data/mnist.npz"
```

This is the MNIST **dataset archive**, not an existing implementation. The data loader reads `x_train`, `y_train`, `x_test`, `y_test` directly and scales uint8 pixels by 255. It never runs Python network requests.

CIFAR-10: default path `../paper1-generalization/data/cifar-10-batches-py/` (unmodified **original 32×32** images, not Paper 1's cropped/whitened transforms). If your archive is elsewhere, pass `--cifar-root PATH` to every command.

## Validate data and architectures first

```powershell
python src/data.py --panel F1
python src/data.py --panel C1
python src/validate.py --panel all --data
```

## Benchmark before authorizing training

Run **on Colab GPU**, once per panel/regime, from repo's `paper2-large-batch/` directory:

```bash
for panel in F1 C1 C2; do
  for regime in SB LB; do
    python src/train.py --mode benchmark --panel "$panel" --regime "$regime" --device cuda --bench-steps 8
  done
done
```

The benchmark prints time per 8 steps, throughput, fetch overhead, estimated **single epoch** duration, and peak allocated GPU memory. It is preliminary (warm-up, eval, BN calibration, Drive time excluded); please send these six outputs for a real total runtime/epoch-budget decision. **Do not launch full runs yet.** Any C2_LB CUDA OOM is important evidence; do not silently switch to gradient accumulation, as this changes BatchNorm statistics.

Windows PowerShell equivalent (GPU only if supported):

```powershell
foreach ($panel in @('F1','C1','C2')) { foreach ($regime in @('SB','LB')) { python src/train.py --mode benchmark --panel $panel --regime $regime --device auto --bench-steps 8 } }
```

Quick single-step sanity test: `python src/train.py --mode smoke --panel F1 --regime SB`.

## Commands reserved for after benchmark approval

Example one run:

```bash
python src/train.py --mode train --panel F1 --regime SB --device cuda --max-epochs 40 --output-root /content/paper2_runs
```

Change `--panel` and `--regime` for all six. `--max-epochs 40` is only a provisional safety cap, **not** an approved or scientifically established convergence budget. Every 5 epochs a fixed first-2048 sample train/test diagnostic is measured, and patience is tested on probe train loss. Full train/test evaluation runs once at the end. Stopping status explicitly distinguishes `loss_plateau` vs `budget_exhausted`. Saved locally per epoch: `last.pt` (weights, Adam, RNG, epoch), final `endpoint.pt` and `metrics.json`. Copy run folders to Drive after checkpoints or at the end; no Drive writes occur in the inner training loop. Re-invoking same run resumes at epoch boundary.

**Current limitations, intentionally disclosed:** This is an MVP, not a fully validated scientific reproduction. The paper omits several architectural specifics; our independent choices are documented in `models.py`. The optimizer uses Adam lr=0.001 as an assumed default, not a confirmed paper hyperparameter. Loss stabilization on a fixed training prefix is an approximate diagnostic; confirm comparable SB/LB train loss and high train accuracy before accepting endpoints. C2_LB may OOM on a T4/A100 due to true 5,000-example BatchNorm batches. DataLoader workers=0 is deliberate because all decoded tensors are cached in RAM, so extra workers may be slower; benchmarking reports batch fetching overhead. No AMP or gradient accumulation is enabled because they can affect methodology.

## After accepted endpoints (not needed yet)

```bash
python src/interpolate.py --panel F1 --device cuda --output-root /content/paper2_runs --bn-policy recalibrate --points 21
```

That script uses all train/test samples for each alpha and outputs measured `F1_interpolation.json` (similarly C1/C2). It interpolates floating-point state entries (trainable weights and, initially, floating buffers). `recalibrate` overwrites BatchNorm running estimates using an explicitly reported 4,096-example training prefix and no gradient updates; `interpolate` uses linearly interpolated floating BN buffers instead. The chosen BN policy is **our reproduction choice**, not confirmed as the paper's. Since the paper's interpolation behavior is sensitive to this decision, choose deliberately; interpolation itself may be expensive and should be separately benchmarked. No plot notebook is included yet, as requested.

## Efficiency revision (2026-10-19)

- Accumulate training loss on the active device, with one scalar synchronization per epoch instead of per batch.
- Write recovery checkpoints every 5 epochs by default (`--checkpoint-every`), plus at stopping or reaching the budget cap.
- Budget exhaustion is **not** tagged as convergence; increasing `--max-epochs` resumes from the checkpoint.
- The 2,048-example fixed probes remain every 5 epochs by default. Final and interpolation metrics use the complete datasets.
- Benchmarks must be executed on CUDA before allocating full budgets; CPU C2 is not a relevant proxy for A100 throughput.
