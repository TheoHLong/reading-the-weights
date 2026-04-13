# Reading the Weights

Task A workspace for the CS7643 final project on bilinear MLP weight interpretability.

Task D work should be logged in `TASK_D_WORKLOG.md`.
Experiment rules live in `RESEARCH_PROTOCOL.md`.
Planned and active runs live in `EXPERIMENT_QUEUE.md`.
Comparable run outcomes live in `RESULTS_TABLE.md`.

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
- `src/reading_weights/data.py`: image dataloaders with train/val/test splits and CIFAR-10 support
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
python scripts/evaluate_truncation.py --checkpoint checkpoints/<best-run>.pt --split test
```

## Workflow

1. Run `python scripts/smoke_test.py --config configs/mnist_baseline.yaml`
2. Train MNIST and Fashion-MNIST with `scripts/train_baseline.py`
3. Export decomposition artifacts with `scripts/analyze_checkpoint.py`
4. Save large artifacts to Drive, not to git

## Environment

Local virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
```

Docker dependency install path:

```bash
docker build -t reading-the-weights .
docker run --rm reading-the-weights python scripts/smoke_test.py --config configs/mnist_baseline.yaml
```

## API reference (for Task B/C/D/E/F)

Below is the interface contract for the core modules. Downstream tasks should only depend on these public APIs.

### model.py

| Symbol | Description |
|--------|-------------|
| `BilinearImageClassifier(d_input, d_hidden, d_output, n_layer, bias, residual, seed)` | The bilinear MLP classifier. `forward(x)` takes `[B, C, H, W]` images and returns `[B, d_output]` logits. |
| `model.embedding_weight` | Property. Returns detached embed weight `[d_hidden, d_input]`. |
| `model.output_weight` | Property. Returns detached head weight `[d_output, d_hidden]`. |
| `model.bilinear_weights` | Property. Returns `[n_layer, 2, d_hidden, d_hidden]` — stacked left/right weight pairs. |
| `build_image_classifier(model_cfg, seed)` | Factory function. Takes the `config['model']` dict and returns a `BilinearImageClassifier`. |

### decomposition.py

| Symbol | Description |
|--------|-------------|
| `build_bilinear_tensor(model)` | Folds the bilinear block and classifier head into one tensor. Returns `[d_output, d_hidden, d_hidden]`. Single-layer models only. |
| `symmetrize_bilinear_tensor(T)` | Returns `(T + T.mT) / 2`. Same shape. |
| `decompose_bilinear_model(model)` | **Main entry point.** Runs the full pipeline and returns a `DecompositionArtifacts` dataclass. |
| `project_eigenvectors_to_input(eigvecs, embed_w)` | Maps eigenvectors from hidden space back to input pixel space. |

**`DecompositionArtifacts` fields (all Tensors):**

| Field | Shape (MNIST example) | Description |
|-------|----------------------|-------------|
| `bilinear_tensor` | `[10, 256, 256]` | Raw Q_c matrices before symmetrization. |
| `symmetrized_tensor` | `[10, 256, 256]` | (Q + Qᵀ)/2 — symmetric matrices for eigendecomposition. |
| `eigenvalues` | `[10, 256]` | Eigenvalues per class, ascending order. **Last = largest.** |
| `eigenvectors_hidden` | `[10, 256, 256]` | Eigenvectors in hidden space. Column `[:,  :, k]` pairs with `eigenvalues[:, k]`. |
| `eigenvectors_input` | `[10, 256, 784]` | Eigenvectors projected to input pixel space. Reshape `[784]` → `[28, 28]` to visualize. |

### data.py

| Symbol | Description |
|--------|-------------|
| `build_image_dataloaders(dataset_cfg, train_cfg)` | Returns a `DatasetBundle(train_loader, val_loader, test_loader, input_shape, num_classes)`. Supports `mnist`, `fashion_mnist`, and `cifar10`. |

### train.py

| Symbol | Description |
|--------|-------------|
| `train_image_experiment(config)` | Runs the full training loop. Returns a dict with paths: `run_dir`, `metrics_path`, `best_checkpoint_path`, `latest_checkpoint_path`. |
| `evaluate(model, loader, criterion, device)` | Returns `(avg_loss, accuracy)`. Already wrapped in `@torch.no_grad()`. |

### utils.py

| Symbol | Description |
|--------|-------------|
| `load_checkpoint(path, map_location='cpu')` | Loads a `.pt` checkpoint. Returns dict with keys: `model_state_dict`, `config`, `epoch`, `metrics`. |
| `resolve_device(name)` | Resolves `'auto'` to `cuda` / `mps` / `cpu`. |
| `set_seed(seed)` | Seeds Python, NumPy, and PyTorch for reproducibility. |

### Checkpoint format

```python
{
    'model_state_dict': OrderedDict,   # pass to model.load_state_dict()
    'config':           dict,          # the full YAML config used for training
    'epoch':            int,           # epoch number when saved
    'metrics':          dict,          # {'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'}
}
```

## Baseline results

| Dataset | Best val acc | Best epoch | Checkpoint |
|---------|-------------|------------|------------|
| MNIST | 97.99% | 38 | `mnist_baseline_20260324-025128.pt` |
| Fashion-MNIST | 89.09% | 94 | `fmnist_baseline_20260324-025914.pt` |

Historical note: these baseline figures were produced before Task D corrected the validation/test split.
