# Paper 2 — Phase 4: Author-code comparison and controlled rerun

**Do not replace the Phase 3 implementation or `results/final_61/`.** This is an independent follow-up experiment performed *after* Phase 3 was committed.

## Reference inspected

- Nitish Keskar and coauthors' repository: <https://github.com/keskarnitish/large-batch-training>
- Architecture definitions: <https://github.com/keskarnitish/large-batch-training/blob/master/network_zoo.py>
- Training and interpolation demonstration: <https://github.com/keskarnitish/large-batch-training/blob/master/plot_parametric_plot.py>
- Preliminary PyTorch demonstration: <https://github.com/keskarnitish/large-batch-training/blob/master/PyTorch/plot_parametric_pytorch.py>

The author's repository includes demonstration code; it is **not guaranteed to be identical to the code used to produce every published figure**. Specifically, the paper Appendix B.3 specifies a convolution stride of **2** for C1 (`[64,5,5,2]`), whereas the Keras `shallownet` uses the default stride **1**. Both implementations are defensible as different specifications; Phase 4 tests the author's released architecture.

## Exact Phase 4 code changes

| Aspect | Phase 3 | Phase 4 | Reason |
|---|---|---|---|
| C1 first and second conv | 5×5, stride 2, padding 2 | 5×5, stride 1, padding 2 | Match Keras `shallownet` implementation; differs from paper text |
| C1 spatial map after pool 2 | 2×2 | 8×8 | Consequence of stride change |
| C1 first dense input | 256 | 4,096 | Required by new map dimensions |
| C1 dropout | 0.5 after both dense hidden layers | Unchanged | Already agreed with author implementation |
| C2 convolutional dropout | None | 0.3 after first 64 conv; 0.4 after early convs of blocks 128/256/512/512 | Match `deepnet` dropout placements |
| C2 classifier dropout | Only after dense512 | Add 0.5 *before* dense512 as well | Match `deepnet` |
| F1 | Phase 3 | Unchanged | No matching author's F1 implementation found in inspected code |
| Other model details | Bias-free convolution / BN-affine convention | Unchanged | Controlled comparison: these may still differ from Keras defaults |
| Adam/batches/seed | Adam 0.001, SB 256, LB 5,000, seed 42 | Unchanged | Isolate architecture as principal change |
| Training data | Raw CIFAR-10 divided by 255 | Unchanged | Match both the experiment and Keras example |
| Interpolation | 61 alpha points, full train/test eval, BN recalibration on 4096 train samples | Unchanged | Controlled Phase 3–4 comparison |

**Caveats:** PyTorch `BatchNorm2d` defaults and bias-free conv layers are *not* bit-for-bit replicas of the Keras implementation, and dropout RNG semantics are different. The author Keras demonstration trains for 20 epochs, whereas this controlled rerun starts with a 50-epoch budget to compare with Phase 3; subsequent training adjustments must be documented rather than retroactively hidden. We make no claim that the author code exactly generated the paper's Figure 3. We do not automatically assume better agreement means a more scientifically valid result.

## Files added (no Phase 3 files modified)

```text
paper2-large-batch/
├── README_PHASE4.md
├── src/
│   ├── phase4_models.py
│   ├── phase4_train.py
│   └── phase4_interpolate.py
└── notebooks/
    └── paper2_phase4_colab.ipynb
```

The single dependency list remains the repository root `requirements.txt`.

## Colab workflow

1. Copy these files into the matching paths in your private repository and commit/push them.
2. Open `paper2-large-batch/notebooks/paper2_phase4_colab.ipynb` in Colab. Enable GPU (A100 recommended) and configure the `GITHUB_TOKEN` secret for cloning your private repo.
3. Run setup, shared root requirements, CIFAR-10 validation, and the **four benchmarks**. **Stop and inspect/send those outputs before committing GPU hours.**
4. Run each C1/C2 SB/LB training cell individually. The first pass is 50 *total* epochs at lr=0.001. Before each restart, run the checkpoint restore cell. Checkpoints are backed up to Drive every 5 epochs; do not remove them from Drive before later stages.
5. Inspect full-dataset training and test endpoints. If training is insufficient, do a transparently recorded additional training stage instead of forcing plots to match.
6. Once endpoints are accepted, run each 61-point interpolation and plot the result. F1 uses the saved Phase 3 JSON from Drive, not newly trained weights.
7. Compare measured Phase 4 endpoints/curves against the saved Phase 3 measurements. State whether each change improves accuracy or qualitative reproduction.

Default local run root: `/content/paper2_phase4_author_aligned/`.
Persistent backup: `/content/drive/MyDrive/ML-Paper-Reproductions/paper2/phase4_author_aligned/`.

Results are **not** hardcoded anywhere. No network weights/checkpoints should be added to Git if they exceed GitHub's size limits; keep optimizer checkpoints in Drive. The three Phase 3 interpolation JSONs remain in `paper2-large-batch/results/final_61/` and on Drive under `paper2/final_61/`.

## Training / interpolation standalone commands (from `paper2-large-batch/`)

```bash
python src/phase4_models.py
python src/phase4_train.py --mode benchmark --panel C1 --regime SB --device cuda
python src/phase4_train.py --mode train --panel C1 --regime SB --device cuda --max-epochs 50 --output-root /content/paper2_phase4_author_aligned --backup-root /content/drive/MyDrive/ML-Paper-Reproductions/paper2/phase4_author_aligned
python src/phase4_interpolate.py --panel C1 --device cuda --points 61 --bn-policy recalibrate --bn-examples 4096 --output-root /content/paper2_phase4_author_aligned
```

## Evidence to retain in Phase 4 report

- Phase 3 commit hash and model selection manifest
- Reference repository URL and paths examined
- The precise architecture differences in the table above
- Per-run training configurations, final checkpoints/metrics, resume stages if any
- GPU benchmark output, train/test accuracies, 61-point interpolation JSONs
- Differences from the paper and from Phase 3 (especially C1 stride conflict)
