# Reading the Weights

Task A workspace for the CS7643 final project on bilinear MLP weight interpretability.

## Scope

This repository currently focuses on Task A:

- implement the bilinear layer forward pass
- implement bilinear tensor construction and symmetrization
- implement the eigendecomposition pipeline
- implement the training framework
- train baseline models on MNIST and Fashion-MNIST

## Layout

- `src/reading_weights/model.py`: bilinear layer and baseline classifier
- `src/reading_weights/decomposition.py`: tensor construction, symmetrization, eigendecomposition
- `src/reading_weights/data.py`: MNIST and Fashion-MNIST dataloaders
- `src/reading_weights/train.py`: training loop and checkpoint logic
- `scripts/`: three runnable entrypoints plus the small path bootstrap helper
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

## Workflow

1. Run `python scripts/smoke_test.py --config configs/mnist_baseline.yaml`
2. Train MNIST and Fashion-MNIST with `scripts/train_baseline.py`
3. Export decomposition artifacts with `scripts/analyze_checkpoint.py`
4. Save large artifacts to Drive, not to git
