# CIFAR Compatibility Worklog

This file records the CIFAR-10 extension and redesign work that started as Task D and now also supports the Task I framing: which modifications actually make CIFAR-10 compatible with weight-based linear analysis.

For pre-existing Task D entries, see the original notes below. New CIFAR compatibility experiments should be logged here.

## 2026-04-22

### Framing update

- Promoted the CIFAR study from a pure Task D workstream to a broader compatibility study that also supports the new Task I framing.
- Working claim: locality-preserving preprocessing changes the spectral behavior of CIFAR-10 in a way that width and scalar regularization do not.
- Added rank-efficiency thresholds as a report-facing metric:
  - minimum rank for `90%` of full-rank accuracy
  - minimum rank for `95%` of full-rank accuracy
  - minimum rank for `99%` of full-rank accuracy

### Code changes

- Added config-controlled L1 regularization support to `src/reading_weights/train.py` via `train.l1_lambda`.
- Extended `scripts/evaluate_truncation.py` to compute and save rank-efficiency thresholds in `test_summary.json`.
- Updated `scripts/analyze_checkpoint.py` and `scripts/evaluate_truncation.py` to load older raw-pixel checkpoints with `strict=False` so pre-preprocess checkpoints remain analyzable after the locality-preprocessing buffer was introduced.
- Added report asset support for the new control and locality-strength runs.
- Added new configs:
  - `configs/cifar10_l1tiny_mps.yaml`
  - `configs/cifar10_locality8_mps.yaml`
  - `configs/cifar10_locality16_mps.yaml`
  - `configs/cifar10_locality16_seed123_mps.yaml`

### Experiment results

- Tiny L1 failed control, matched 25-epoch budget
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/cifar10_l1tiny_mps.yaml`
  - Run: `cifar10_l1tiny_mps_20260422-203929`
  - Best val accuracy: `0.4260`
  - Test accuracy: `0.4180`
  - Rank-efficiency: `90% -> 32`, `95% -> 64`, `99% -> 128`
  - Interpretation: tiny L1 does not improve top-line performance or spectral concentration over the raw 25-epoch baseline.

- Larger locality variant, `8x8` pooling
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/cifar10_locality8_mps.yaml`
  - Run: `cifar10_locality8_mps_20260422-204446`
  - Best val accuracy: `0.3826`
  - Test accuracy: `0.3812`
  - Rank-efficiency: `90% -> 16`, `95% -> 32`, `99% -> 32`
  - Interpretation: stronger pooling increases compressibility, but gives up too much CIFAR signal compared with `4x4`.

- Larger locality variant, `16x16` pooling
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/cifar10_locality16_mps.yaml`
  - Run: `cifar10_locality16_mps_20260422-204115`
  - Best val accuracy: `0.2604`
  - Test accuracy: `0.2556`
  - Rank-efficiency: `90% -> 8`, `95% -> 8`, `99% -> 16`
  - Interpretation: `16x16` pooling overcompresses the image and collapses task performance.

- `16x16` pooling replication
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/cifar10_locality16_seed123_mps.yaml`
  - Run: `cifar10_locality16_seed123_mps_20260422-204255`
  - Best val accuracy: `0.2518`
  - Test accuracy: `0.2570`
  - Rank-efficiency: `90% -> 8`, `95% -> 8`, `99% -> 16`
  - Interpretation: the overcompression failure is replicated and not a seed artifact.

### Current read

- Raw-pixel CIFAR remains broad-spectrum.
- Width, lower weight decay, and tiny L1 are failed controls.
- `4x4` pooling remains the best locality-preserving regime found so far.
- `8x8` pooling supports the idea of a locality-strength tradeoff.
- `16x16` pooling shows that extreme pooling can produce a low-rank but scientifically unhelpful classifier.
- The strongest paper story is now:
  - MNIST low-rank
  - raw CIFAR broad-spectrum
  - scalar controls fail
  - moderate locality (`4x4`) restores compatibility
  - excessive locality (`16x16`) overcompresses the task

## Legacy record

The original Task D worklog content is preserved in `TASK_D_WORKLOG.md`.

## 2026-04-27

### Calibration against Task G results

Kai reported provisional Task G validation accuracies:

- Raw `n_layer=1` bilinear student: about `44.9%`
- Knowledge distillation: about `45.1%`
- CKA guidance: about `46.6%`

Goal for this pass: test whether locality-preserving pooling can meet the raw-student baseline and approach the transfer-guided result without changing the model family.

### New configs

- `configs/cifar10_locality4_50e_mps.yaml`
- `configs/cifar10_locality4_wd001_50e_mps.yaml`
- `configs/cifar10_locality2_50e_mps.yaml`

### Results

- `4x4` pooling, 50 epochs, original weight decay
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/cifar10_locality4_50e_mps.yaml`
  - Run: `cifar10_locality4_50e_mps_20260427-224921`
  - Best validation accuracy: `0.4574`
  - Best validation loss: `1.662405`
  - Test accuracy at best validation checkpoint: `0.4612`
  - Rank-efficiency: `90% -> 32`, `95% -> 32`, `99% -> 64`
  - Interpretation: simply training the best pooling regime longer clears Kai's raw-student baseline and keeps the spectral advantage.

- `4x4` pooling, 50 epochs, lower weight decay
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/cifar10_locality4_wd001_50e_mps.yaml`
  - Run: `cifar10_locality4_wd001_50e_mps_20260427-225245`
  - Best validation accuracy: `0.4592`
  - Best validation loss: `1.658855`
  - Test accuracy at best validation checkpoint: `0.4638`
  - Rank-efficiency: `90% -> 32`, `95% -> 32`, `99% -> 64`
  - Interpretation: lower weight decay gives a small accuracy gain while preserving the same rank-efficiency profile. This is the current best pooling result.

- `2x2` pooling, 50 epochs, original weight decay
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/cifar10_locality2_50e_mps.yaml`
  - Run: `cifar10_locality2_50e_mps_20260427-225510`
  - Best validation accuracy: `0.4566`
  - Best validation loss: `1.703159`
  - Test accuracy at best validation checkpoint: `0.4509`
  - Rank-efficiency: `90% -> 64`, `95% -> 64`, `99% -> 128`
  - Interpretation: preserving more spatial detail does not improve validation enough to justify the broader spectrum.

### Current read

- Pooling now meets the raw-student Task G baseline on validation.
- The best pooled model is close to, but does not yet meet, Kai's provisional CKA-guided `46.6%` validation result.
- The strongest comparison is now:
  - raw-pixel 50e: `44.52%` validation, `44.08%` test, `99%` rank-efficiency at rank `128`
  - `4x4` pooled 50e wd=0.01: `45.92%` validation, `46.38%` test, `99%` rank-efficiency at rank `64`
- This supports the claim that locality-preserving preprocessing gives accuracy gains comparable to modest transfer gains while also improving compressibility.
