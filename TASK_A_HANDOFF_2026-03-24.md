# Task A Handoff

## 1. Status

Task A is complete and stable.
The following pipeline has been run end-to-end successfully:

- smoke test
- MNIST baseline training
- MNIST checkpoint decomposition
- Fashion-MNIST baseline training
- Fashion-MNIST checkpoint decomposition

Known-good baseline results:
- MNIST best val_acc = 0.9799
- Fashion-MNIST best val_acc = 0.8909

## 2. Stable base

Please branch from the latest `main` after the handoff commit lands.
If you need the exact pre-handoff-doc baseline, use:

- commit: `e9124bc791950a77b65c1e82c3f83b017b772957`

Core repo documentation:
- scope and artifact contract: `README.md`
- API reference: `README.md`
- training entrypoint: `scripts/train_baseline.py`
- decomposition entrypoint: `scripts/analyze_checkpoint.py`

## 3. Stable artifacts

Known-good checkpoints:
- `checkpoints/mnist_baseline_20260324-025128.pt`
- `checkpoints/fmnist_baseline_20260324-025914.pt`

Known-good analysis outputs:
- `results/analysis/mnist_baseline_20260324-025128/decomposition.pt`
- `results/analysis/mnist_baseline_20260324-025128/summary.json`
- `results/analysis/fmnist_baseline_20260324-025914/decomposition.pt`
- `results/analysis/fmnist_baseline_20260324-025914/summary.json`

Training metrics:
- `results/metrics/mnist_baseline_20260324-025128/metrics.csv`
- `results/metrics/fmnist_baseline_20260324-025914/metrics.csv`

## 4. Public interfaces you should rely on

Model:
- `src/reading_weights/model.py`
- `build_image_classifier(model_cfg, seed)`

Training:
- `src/reading_weights/train.py`
- `train_image_experiment(config)`

Decomposition:
- `src/reading_weights/decomposition.py`
- `decompose_bilinear_model(model)`

Data:
- `src/reading_weights/data.py`
- `build_image_dataloaders(dataset_cfg, train_cfg)`

## 5. Ownership boundaries

Module B:
- own training-side extensions and metric-side analysis
- consume checkpoints and metrics outputs
- do not change model/decomposition interfaces unless necessary

Module C:
- own decomposition-side analysis and visualization
- consume `decomposition.pt` and `summary.json`
- do not change training/checkpoint interfaces unless necessary

## 6. Tensor semantics for C

Important shapes in `decomposition.pt`:
- `eigenvalues`: `[10, 256]`
- `eigenvectors_hidden`: `[10, 256, 256]`
- `eigenvectors_input`: `[10, 256, 784]`

Interpretation:
- `eigenvectors_input[class_idx, component_idx]` is a length-784 vector
- reshape it to `28 x 28` for visualization
- eigenvalues are in ascending order, so the last index is the largest eigenvalue

## 7. Current limitations

Current Task A code only supports:
- MNIST
- Fashion-MNIST
- single-layer bilinear baseline for decomposition

It is not the place to add CIFAR or transfer-learning logic.

## 8. Collaboration rule

If either of you needs to modify:
- checkpoint format
- model public properties
- decomposition output structure

please notify the group first, because those are shared interfaces.
