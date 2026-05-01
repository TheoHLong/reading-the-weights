# Reading the Weights

CS7643 final project workspace for bilinear MLP weight interpretability.

The project narrative is now:

1. Reproduce the bilinear MLP weight-reading pipeline on MNIST and Fashion-MNIST.
2. Extend the same pipeline to CIFAR-10 as a harder natural-image stress test.
3. Study whether KD or CKA transfer changes bilinear student training behavior and weight structure.
4. Add Signed Quadratic Shrink (SQS) gates as a stability/interpretability extension for bilinear-style GLUs.

## Layout

- `src/`: importable Python package.
- `scripts/`: runnable entrypoints.
- `configs/`: experiment configs.
- `notebooks/`: Colab/local notebooks.
- `docs/handoffs/`: task handoff notes.
- `docs/plans/`: implementation plans and experiment notes.
- `checkpoints/`: generated model checkpoints, ignored by git except `.gitkeep`.
- `results/`: generated metrics, analysis artifacts, adversarial outputs, and diagnostics, ignored by git except `.gitkeep`.

## Main Commands

Use the project environment that has PyTorch installed. On the current machine:

```bash
/Users/longtenghai/opt/anaconda3/envs/web-env/bin/python scripts/check_cka.py
```

Core replication:

```bash
python scripts/smoke_test.py --config configs/baselines/mnist_baseline.yaml
python scripts/train_baseline.py --config configs/baselines/mnist_baseline.yaml
python scripts/train_baseline.py --config configs/baselines/fmnist_baseline.yaml
python scripts/analyze_checkpoint.py --checkpoint checkpoints/<best-run>.pt
```

SQS paper-style MNIST/Fashion-MNIST runs:

```bash
python scripts/train_baseline.py --config configs/baselines/mnist_paper_bilinear.yaml
python scripts/train_baseline.py --config configs/baselines/mnist_paper_sqs.yaml
python scripts/train_baseline.py --config configs/baselines/fmnist_paper_bilinear.yaml
python scripts/train_baseline.py --config configs/baselines/fmnist_paper_sqs.yaml
```

CIFAR-10 baseline and teacher:

```bash
python scripts/train_baseline.py --config configs/baselines/cifar10_baseline.yaml
python scripts/train_guide.py --config configs/guides/resnet18_cifar10.yaml
python scripts/eval_checkpoint.py --checkpoint checkpoints/resnet18_cifar10_teacher.pt
```

Transfer runs:

```bash
python scripts/train_transfer.py --config configs/transfer/cifar10_kd.yaml
python scripts/train_transfer.py --config configs/transfer/cifar10_cka.yaml
python scripts/train_transfer.py --config configs/transfer/cifar10_cka_n4.yaml
```

Adversarial masks:

```bash
python scripts/run_adversarial.py --checkpoint checkpoints/mnist_baseline_20260324-025128.pt
python scripts/run_adversarial.py --checkpoint checkpoints/fmnist_baseline_20260324-025914.pt
python scripts/run_adversarial.py --checkpoint checkpoints/cifar10_baseline_20260427-215646.pt
```

SQS eigenspectrum comparison:

```bash
python scripts/analyze_checkpoint.py --checkpoint checkpoints/mnist_paper_bilinear_<timestamp>.pt
python scripts/analyze_checkpoint.py --checkpoint checkpoints/mnist_paper_sqs_<timestamp>.pt
python scripts/compare_eigenvectors.py \
  --bilinear-decomps results/analysis/mnist_paper_bilinear_<timestamp>/decomposition.pt \
  --sqs-decomps results/analysis/mnist_paper_sqs_<timestamp>/decomposition.pt \
  --output-dir results/figures/mnist_paper_sqs_vs_bilinear_figure_2b
python scripts/visualize_eigenvectors.py \
  --decomposition results/analysis/mnist_paper_sqs_<timestamp>/decomposition.pt \
  --image-size 28 --channels 1 --label-set mnist \
  --top-k 3 --bottom-k 3 \
  --output results/figures/mnist_paper_sqs_eigenvectors.png
```

## Current Results

| Experiment | Best val acc | Notes |
| --- | ---: | --- |
| MNIST bilinear baseline | 97.99% | Core replication checkpoint exists. |
| Fashion-MNIST bilinear baseline | 89.09% | Core replication checkpoint exists. |
| MNIST paper-style bilinear / SQS | 95.38% / 95.20% | Uses SQS paper hyperparameters: noise=1.0, wd=0.1, batch=2048, epochs=20. |
| Fashion-MNIST paper-style bilinear / SQS | 78.43% / 82.72% | SQS improves this local paper-style run; bilinear appears under-converged. |
| CIFAR-10 1-layer bilinear baseline | 44.88% | Raw-pixel natural-image stress test. |
| CIFAR-10 1-layer SQS baseline / CKA | 47.22% / 47.70% | SQS gives a small single-layer gain; CKA gain is modest. |
| CIFAR-10 ResNet-18 guide | 95.98% | Teacher checkpoint alias: `checkpoints/resnet18_cifar10_teacher.pt`. |
| CIFAR-10 KD student | 45.10% | KD gives little improvement over 1-layer baseline. |
| CIFAR-10 1-layer CKA student | 46.62% | Small but measurable improvement. |
| CIFAR-10 4-layer pure bilinear baseline / CKA | 42.50% / 46.76% | Both collapse after early epochs in the latest run; deep pure bilinear is unstable. |
| CIFAR-10 4-layer SQS baseline / CKA | 57.96% / 58.70% | SQS stabilizes depth; CKA adds a small extra gain. |
| CIFAR-10 4-layer gated CKA student | 61.60% | Historical `gate: silu` run; useful as an optimization reference, not a bilinear decomposition target. |

## Artifact Contract

Training runs write:

- `checkpoints/<run_name>.pt`
- `checkpoints/<run_name>_latest.pt`
- `results/metrics/<run_name>/metrics.csv`
- `results/metrics/<run_name>/summary.json`

Analysis runs write:

- `results/analysis/<checkpoint_name>/decomposition.pt`
- `results/analysis/<checkpoint_name>/summary.json`
- `results/figures/<run_name>/...` for generated SQS comparison plots and CSVs

Adversarial runs write:

- `results/adversarial/<checkpoint_name>/fig7_panel.png`
- `results/adversarial/<checkpoint_name>/success_curves.png`
- `results/adversarial/<checkpoint_name>/masks.png`
- `results/adversarial/<checkpoint_name>/attack_metrics.json`
- `results/adversarial/<checkpoint_name>/masks.pt`

## Public APIs

### `src.model`

| Symbol | Description |
| --- | --- |
| `BilinearImageClassifier(...)` | Bilinear MLP image classifier. `forward(x)` accepts `[B, C, H, W]` images and returns logits. |
| `build_image_classifier(model_cfg, seed)` | Factory for configs. |
| `SignedQuadraticShrink(...)` | SQS gate with paper defaults `c=0.01`, `lambda=0.5`, used by `gate: sqs`. |
| `model.embedding_weight` | Detached embedding weight `[d_hidden, d_input]`. |
| `model.output_weight` | Detached head weight `[d_output, d_hidden]`. |
| `model.bilinear_weights` | Stacked left/right bilinear weights `[n_layer, 2, d_hidden, d_hidden]`. |

### `src.decomposition`

| Symbol | Description |
| --- | --- |
| `build_bilinear_tensor(model)` | Builds class tensors `[d_output, d_hidden, d_hidden]`. Single-layer `gate=None` and paper-style `gate='sqs'` models only. |
| `symmetrize_bilinear_tensor(T)` | Returns `(T + T.mT) / 2`. |
| `decompose_bilinear_model(model)` | Main decomposition entrypoint. |
| `project_eigenvectors_to_input(eigvecs, embed_w)` | Projects hidden-space eigenvectors back to pixels. |

### `src.data`

| Symbol | Description |
| --- | --- |
| `build_image_dataloaders(dataset_cfg, train_cfg)` | Returns train/val/test loaders plus shape metadata. Supports `mnist`, `fashion_mnist`, and `cifar10`. |
| `CIFAR10_MEAN`, `CIFAR10_STD` | Shared CIFAR-10 normalization constants for student and guide models. |

### `src.transfer`

| Symbol | Description |
| --- | --- |
| `kd_loss(...)` | Knowledge-distillation objective. |
| `train_kd_experiment(config)` | Train a bilinear student with teacher logits. |
| `train_cka_experiment(config)` | Train a bilinear student with CKA representational alignment. |
| `cka_loss(...)` | Layer-pair CKA distance reducer. |

### `src.adversarial`

| Symbol | Description |
| --- | --- |
| `compute_adversarial_masks(...)` | Builds Fig. 7-style pseudoinverse masks. |
| `compute_permuted_masks(...)` | Random-permutation baseline preserving mask distribution and norm. |
| `evaluate_attacks(...)` | Sweeps perturbation magnitudes and reports accuracy and target-hit rates. |

## Limitations

- Decomposition currently supports single-layer `gate=None` and `gate='sqs'` students only. Deep checkpoints can be evaluated, but not decomposed yet.
- SQS decomposition follows the SQS paper's weight-spectrum approximation: it forms interaction matrices directly from the GLU weights. It is not an exact pure quadratic model.
- ReLU/GELU/SiLU checkpoints are excluded from weight-space bilinear decomposition because those gates do not preserve this approximation.
- The previous strongest 4-layer CKA result used `gate: silu`, so it should be reported only as a training/optimization observation.
- Random/noise guide variants also perform well, so CKA results should be interpreted as representational regularization evidence, not as a clean claim that trained ResNet semantics transferred.
