# Reading the Weights

CS7643 final project repository for bilinear MLP weight interpretability.

The final report asks whether the weight-reading framework of Pearce et al.
scales beyond toy grayscale data. We reproduce the single-layer bilinear
decomposition pipeline on MNIST/Fashion-MNIST, stress-test it on CIFAR-10,
and test whether ResNet-derived KD/CKA alignment can supply the missing
inductive bias. The main finding is structural: bilinear weight-reading applies
and trains stably only in the single-layer regime. Locality helps on CIFAR-10,
but depth removes the per-class `Q_c` matrix readout and introduces a
degree-`2^L` optimization pathology.

## Final Submission

The final report is maintained and submitted separately through Overleaf. This
GitHub repository contains the implementation, configs, notebooks, and
supporting generated figures/tables needed to understand the experiments.

LaTeX source, Overleaf upload archives, and LaTeX build byproducts are
intentionally ignored by git. The repository is meant to contain source code,
configs, and lightweight supporting assets, not generated checkpoints or local
experiment dumps.

## Layout

- `src/`: importable Python package.
- `scripts/`: runnable entrypoints.
- `configs/`: experiment configs.
- `notebooks/`: Colab/local notebooks.
- `docs/handoffs/`: task handoff notes.
- `docs/plans/`: implementation plans and experiment notes.
- `report_assets/`: checked-in report figures, summary tables, and narrative snippets.
- `checkpoints/`: generated model checkpoints, ignored by git except `.gitkeep`.
- `results/`: generated metrics, analysis artifacts, adversarial outputs, and diagnostics, ignored by git except `.gitkeep`.

## Reproducing Report Runs

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

Core replication:

```bash
python scripts/smoke_test.py --config configs/baselines/mnist_baseline.yaml
python scripts/train_baseline.py --config configs/baselines/mnist_baseline.yaml
python scripts/train_baseline.py --config configs/baselines/fmnist_baseline.yaml
python scripts/analyze_checkpoint.py --checkpoint checkpoints/<best-run>.pt
```

CIFAR-10 locality runs:

```bash
python scripts/train_baseline.py --config configs/cifar_compatibility/cifar10_foundation_baseline.yaml
python scripts/train_baseline.py --config configs/cifar_compatibility/cifar10_locality2_50e_mps.yaml
python scripts/train_baseline.py --config configs/cifar_compatibility/cifar10_locality4_wd001_50e_mps.yaml
python scripts/evaluate_truncation.py --checkpoint checkpoints/<best-run>.pt --split test
```

Transfer and guide runs:

```bash
python scripts/train_guide.py --config configs/guides/resnet18_cifar10.yaml
python scripts/train_transfer.py --config configs/transfer/cifar10_kd.yaml
python scripts/train_transfer.py --config configs/transfer/cifar10_cka.yaml
python scripts/train_transfer.py --config configs/transfer/cifar10_cka_n4.yaml
python scripts/train_transfer.py --config configs/transfer/cifar10_cka_n4_random_cnn_auto5.yaml
python scripts/train_baseline.py --config configs/baselines/cifar10_baseline_n4_silu_s44.yaml
python scripts/train_transfer.py --config configs/transfer/cifar10_cka_n4_silu_s44.yaml
```

Report asset builders:

```bash
python scripts/noise_sweep_analysis.py --config configs/baselines/mnist_baseline.yaml --mode norm
python scripts/train_with_dynamics.py --config configs/baselines/mnist_baseline.yaml --checkpoint-after 10
python scripts/run_adversarial.py --checkpoint checkpoints/mnist_baseline_20260324-025128.pt
python scripts/run_adversarial.py --checkpoint checkpoints/fmnist_baseline_20260324-025914.pt
python scripts/build_task_c_report_assets.py --mnist-decomposition results/analysis/<mnist-run>/decomposition.pt --fmnist-decomposition results/analysis/<fmnist-run>/decomposition.pt
python scripts/build_task_d_report_assets.py
```

Note: the two scripts apply input noise with different semantics on purpose.

- `noise_sweep_analysis.py` is an input-perturbation experiment. On unit-range
  datasets (MNIST / Fashion-MNIST) it clamps noisy inputs back to [0, 1] so
  each sample stays a valid image; CIFAR-10 (already normalized) is left
  unclamped. Control via `--clamp {auto,unit,none}`.
- `train_with_dynamics.py` reuses the project's existing `train.input_noise_std`
  training-noise path and does **not** clamp by default. This preserves the
  semantics of existing configs that train with input noise as regularization.
  Pass `--clamp-noisy-inputs` (or set
  `train.clamp_noisy_inputs: true` in the config) to match the Task B
  perturbation semantics.

## Reported Results

| Experiment | Val | Test | Notes |
| --- | ---: | ---: | --- |
| MNIST 1-layer bilinear | -- | 97.99% | Core replication. |
| Fashion-MNIST 1-layer bilinear | -- | 89.09% | Core replication. |
| MNIST regularized figure run | -- | 98.16% | Used for eigenvector figures. |
| Fashion-MNIST regularized figure run | -- | 87.13% | Used for eigenvector figures. |
| CIFAR-10 raw-pixel 50e | -- | 44.52% | `r_99=128`, broad spectrum. |
| CIFAR-10 4x4 locality 50e | -- | 46.12% | `r_99=64`, sharper spectrum. |
| ResNet-18 guide | 95.98% | 95.08% | External reference network. |
| 1L bilinear, no transfer | 44.20% | 43.86% | Exact readout. |
| 1L bilinear + KD | 44.58% | 44.26% | Little change. |
| 1L bilinear + CKA | 46.20% | 46.40% | Small gain. |
| 4L bilinear, no CKA | 42.50% | 42.79% | Unstable depth run. |
| 4L bilinear + CKA | 52.38% | 52.23% | Helps training, not exact single-layer readout. |
| 4L bilinear + random-guide CKA | 50.74% | -- | Validation-only diagnostic. |
| 4L SiLU, no CKA | 46.88% | 48.20% | Non-decomposable control. |
| 4L SiLU + CKA | 61.02% | 60.67% | Strong apparent CKA response, confounded with gate/capacity. |

## Artifact Policy

Training runs write:

- `checkpoints/<run_name>.pt`
- `checkpoints/<run_name>_latest.pt`
- `results/metrics/<run_name>/metrics.csv`
- `results/metrics/<run_name>/summary.json`

Analysis runs write:

- `results/analysis/<checkpoint_name>/decomposition.pt`
- `results/analysis/<checkpoint_name>/summary.json`
- `results/truncation/<checkpoint_name>/<split>_truncation.csv`
- `results/truncation/<checkpoint_name>/<split>_summary.json`
- `results/figures/<run_name>/...` for generated plots and CSVs

Adversarial runs write:

- `results/adversarial/<checkpoint_name>/fig7_panel.png`
- `results/adversarial/<checkpoint_name>/success_curves.png`
- `results/adversarial/<checkpoint_name>/masks.png`
- `results/adversarial/<checkpoint_name>/attack_metrics.json`
- `results/adversarial/<checkpoint_name>/masks.pt`

These generated outputs are ignored by git because they are large and
machine-specific. The repository keeps configs, code, and lightweight
supporting figures/tables instead.

## Public APIs

### `src.model`

| Symbol | Description |
| --- | --- |
| `BilinearImageClassifier(...)` | Bilinear MLP image classifier. `forward(x)` accepts `[B, C, H, W]` images and returns logits. |
| `build_image_classifier(model_cfg, seed)` | Factory for configs. |
| `build_input_projection(...)` | Builds fixed input-side projections such as non-trainable average pooling for CIFAR compatibility runs. |
| `load_image_classifier_state(model, state_dict)` | Loads student checkpoints while tolerating the deterministic `input_projection` buffer added after older raw-pixel runs. |
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
| `spectral_effective_rank(eigenvalues)` | Entropy effective rank per class, used for training dynamics and noise sweep summaries. |

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

- Exact per-class `Q_c` decomposition applies only to single-layer bilinear students. Deep bilinear checkpoints can be evaluated, but the report treats them as outside the exact matrix-readout framework.
- SQS utilities remain in the codebase from exploratory runs, but SQS is not part of the final report's main claim.
- ReLU/GELU/SiLU checkpoints are excluded from exact weight-space bilinear decomposition because those gates do not preserve the quadratic form.
- The strongest 4-layer CKA result uses `gate: silu`, so it is reported only as a training/optimization diagnostic.
- Random/noise guide variants also perform well, so CKA results should be interpreted as representational regularization evidence, not as a clean claim that trained ResNet semantics transferred.
