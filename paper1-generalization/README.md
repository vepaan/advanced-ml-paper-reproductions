# Paper 1 — Rethinking Generalization

Reproduction of Figure 1(a), 1(b), and 1(c) from Zhang et al. (2017), *Understanding Deep Learning Requires Rethinking Generalization*, for CS 6362 Advanced Machine Learning.

The experiments investigate how neural networks fit randomized labels and inputs, and how label corruption affects convergence time and generalization.

## Repository Structure

```text
paper1-generalization/
├── src/
│   ├── data.py                 # CIFAR-10 preprocessing and randomization
│   ├── models.py               # MLP, AlexNet, and Small Inception
│   ├── train.py                # Training, evaluation, and checkpointing
│   ├── experiments.py          # Experiment orchestration (indices 1–27)
│   ├── plot.py                 # Standalone plotting utility
│   └── ...                     # Architecture audits and helper scripts
├── notebooks/
│   ├── reproduce_figure1.ipynb
│   └── reproduce_figure1_threshold_1pct.ipynb
├── results/
│   └── figure1/                # Training logs, metadata, and summaries
└── data/                       # CIFAR-10 dataset (downloaded locally)
```

## Experiments

- **Figure 1(a):** Training loss versus optimization steps for Inception under original labels, random labels, shuffled pixels, and Gaussian inputs.
- **Figure 1(b):** Relative convergence time as label corruption increases.
- **Figure 1(c):** Test error as label corruption increases.

The label-corruption experiments use probabilities **0, 0.2, 0.6, and 1.0**, across MLP, AlexNet, and Small Inception.

Training uses PyTorch, SGD with momentum, and the learning-rate schedule described in the paper. Experiment results are stored as CSV logs with accompanying metadata.

## Reproducing the Figures

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open either notebook in `notebooks/` and run all cells. The notebooks read the existing data from `results/figure1/` and display the figures inline. **No retraining is required.**

- `reproduce_figure1.ipynb` uses the original 0.1% convergence-error threshold.
- `reproduce_figure1_threshold_1pct.ipynb` retrospectively evaluates a 1% threshold using the same training logs.

## Notes and Limitations

- MLP and AlexNet match the paper's reported parameter counts. Our Small Inception implementation has approximately 0.74% more parameters.
- The original 0.1% convergence criterion was not reached for several corrupted-label MLP experiments. The 1% sensitivity analysis provides convergence measurements for these runs without additional training.
- The reduced corruption grid and other implementation differences are documented in the accompanying report.
- The Phase 3 implementation was developed independently. Phase 4 compares it with the author's publicly released demonstration code.

**Reference:** Zhang et al., *Understanding Deep Learning Requires Rethinking Generalization*, ICLR 2017.