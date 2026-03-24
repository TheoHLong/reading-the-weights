# Reading the Weights

Task A workspace for the CS7643 final project on bilinear MLP weight interpretability.

## Scope

This repository currently focuses on Task A only:

- implement the bilinear layer forward pass
- implement bilinear tensor construction and symmetrization
- implement the eigendecomposition pipeline
- implement the training framework
- train baseline models on MNIST and Fashion-MNIST

Anything outside Task A stays out of this code path until the baseline artifacts are stable.

## A-only layout

- `src/reading_weights/models/`: bilinear layer and baseline classifier
- `src/reading_weights/analysis/`: tensor construction, symmetrization, eigendecomposition
- `src/reading_weights/data/`: MNIST and Fashion-MNIST dataloaders
- `scripts/train_baseline.py`: train a baseline from a YAML config
- `scripts/analyze_checkpoint.py`: export decomposition artifacts from a checkpoint
- `configs/`: MNIST and Fashion-MNIST baseline configs

## Artifact contract

Each training run should produce:

- a best checkpoint in `checkpoints/`
- a latest checkpoint in `checkpoints/`
- per-epoch metrics in `results/metrics/<run_name>/metrics.csv`
- a run summary in `results/metrics/<run_name>/summary.json`

Each analysis run should produce:

- `results/analysis/<checkpoint_name>/decomposition.pt`
- `results/analysis/<checkpoint_name>/summary.json`

## Main commands

```bash
python scripts/smoke_test.py --config configs/mnist_baseline.yaml
python scripts/train_baseline.py --config configs/mnist_baseline.yaml
python scripts/train_baseline.py --config configs/fmnist_baseline.yaml
python scripts/analyze_checkpoint.py --checkpoint checkpoints/<best-run>.pt
```

## Colab workflow

1. Push this repository to GitHub.
2. Open Colab and mount Google Drive.
3. Clone the repo into the Colab runtime.
4. Run `bash scripts/setup_colab.sh`.
5. Run one baseline config at a time.
6. Save large artifacts to Drive, not to git.
