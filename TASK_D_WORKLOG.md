# Task D Worklog

This file records all Task D changes to Kai's code, all new code additions, environment setup, experiment setup, and results.

## 2026-04-12

### Scope guardrails

- Task D remains focused on a course-valid Extension 1 path: CIFAR-10 support, valid evaluation protocol, and decomposition-ready baselines.
- The larger redesign question is being kept explicit: determine whether weight-based decomposition survives on CIFAR-10 only after introducing the minimum necessary experimental rigor.

### Environment setup

- Created a local `uv` virtual environment at `.venv` with `uv venv .venv`.
- Installed local experiment dependencies with `uv pip install --python .venv/bin/python -e . -r requirements.txt`.
- Standardized the repo on a `uv`-based workflow for experiments.
- Added a `Dockerfile` so dependencies can be installed inside Docker as required.
- Added `.dockerignore` to keep large artifacts and local environments out of image builds.
- Created feature branch `task-c-cifar10-foundation` to keep all Task D work off `main`.

### Code changes to Kai's base

- Extended dataset support from MNIST/Fashion-MNIST to include CIFAR-10.
- Added explicit train/validation/test separation so checkpoint selection no longer uses the test split.
- Added configurable train-time augmentation and normalization hooks in the dataset builder.
- Added a CIFAR-10 baseline config with normalization and lightweight augmentation.
- Added pilot configs for short local CIFAR-10 execution on CPU and MPS.
- Added a longer headline-style CIFAR-10 MPS config for stronger local evidence.
- Added an MNIST MPS pilot config for direct truncation comparison against CIFAR-10.
- Added final test-set evaluation for the checkpoint selected by validation accuracy.
- Added a CPU-safe macOS fallback in training to avoid `torch_shm_manager` worker crashes during local runs.
- Added `scripts/evaluate_truncation.py` to measure eigenvalue truncation vs. accuracy directly from checkpoints.

### New files

- `configs/cifar10_baseline.yaml`
- `configs/cifar10_pilot.yaml`
- `configs/cifar10_pilot_mps.yaml`
- `configs/cifar10_headline_mps.yaml`
- `configs/mnist_pilot_mps.yaml`
- `Dockerfile`
- `.dockerignore`
- `TASK_D_WORKLOG.md`
- `RESEARCH_PROTOCOL.md`
- `EXPERIMENT_QUEUE.md`
- `RESULTS_TABLE.md`
- `scripts/evaluate_truncation.py`

### Results

- CIFAR-10 smoke test, local `uv` environment, config `configs/cifar10_baseline.yaml`
  - Command: `.venv/bin/python scripts/smoke_test.py --config configs/cifar10_baseline.yaml`
  - Hardware: `mps`
  - Run name: `cifar10_baseline_smoke_20260412-221302`
  - Best val accuracy: `0.140625`
  - Test accuracy at best validation checkpoint: `0.140625`
  - Artifacts:
    - `results/metrics/cifar10_baseline_smoke_20260412-221302/metrics.csv`
    - `checkpoints/cifar10_baseline_smoke_20260412-221302.pt`
    - `results/analysis/cifar10_baseline_smoke_20260412-221302/decomposition.pt`

- CIFAR-10 CPU pilot, local `uv` environment, config `configs/cifar10_pilot.yaml`
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/cifar10_pilot.yaml`
  - Hardware: `cpu`
  - Run name: `cifar10_pilot_20260412-221341`
  - Epochs: `3`
  - Best val accuracy: `0.3588`
  - Test accuracy at best validation checkpoint: `0.3523`
  - Analysis command: `.venv/bin/python scripts/analyze_checkpoint.py --checkpoint checkpoints/cifar10_pilot_20260412-221341.pt`
  - Top eigenvalue summary: class `1`, eigenvalue `0.016802817583084106`
  - Artifacts:
    - `results/metrics/cifar10_pilot_20260412-221341/metrics.csv`
    - `results/analysis/cifar10_pilot_20260412-221341/decomposition.pt`

- CIFAR-10 MPS pilot, local `uv` environment, config `configs/cifar10_pilot_mps.yaml`
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/cifar10_pilot_mps.yaml`
  - Hardware: `mps`
  - Run name: `cifar10_pilot_mps_20260412-221450`
  - Epochs: `10`
  - Best val accuracy: `0.3942`
  - Test accuracy at best validation checkpoint: `0.3967`
  - Analysis command: `.venv/bin/python scripts/analyze_checkpoint.py --checkpoint checkpoints/cifar10_pilot_mps_20260412-221450.pt`
  - Top eigenvalue summary: class `6`, eigenvalue `0.014459285885095596`
  - Artifacts:
    - `results/metrics/cifar10_pilot_mps_20260412-221450/metrics.csv`
    - `results/analysis/cifar10_pilot_mps_20260412-221450/decomposition.pt`

- MNIST MPS pilot, local `uv` environment, config `configs/mnist_pilot_mps.yaml`
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/mnist_pilot_mps.yaml`
  - Hardware: `mps`
  - Run name: `mnist_pilot_mps_20260412-223552`
  - Epochs: `10`
  - Best val accuracy: `0.9671666666666666`
  - Test accuracy at best validation checkpoint: `0.9684`
  - Analysis command: `.venv/bin/python scripts/analyze_checkpoint.py --checkpoint checkpoints/mnist_pilot_mps_20260412-223552.pt`
  - Truncation command: `.venv/bin/python scripts/evaluate_truncation.py --checkpoint checkpoints/mnist_pilot_mps_20260412-223552.pt --split test --device mps`
  - Top eigenvalue summary: class `1`, eigenvalue `0.256197065114975`
  - Key truncation points:
    - rank `8` accuracy `0.9640`
    - rank `16` accuracy `0.9686`
    - full rank `256` accuracy `0.9684`

- CIFAR-10 headline MPS baseline, local `uv` environment, config `configs/cifar10_headline_mps.yaml`
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/cifar10_headline_mps.yaml`
  - Hardware: `mps`
  - Run name: `cifar10_headline_mps_20260412-223650`
  - Epochs: `25`
  - Best val accuracy: `0.4258`
  - Test accuracy at best validation checkpoint: `0.4181`
  - Analysis command: `.venv/bin/python scripts/analyze_checkpoint.py --checkpoint checkpoints/cifar10_headline_mps_20260412-223650.pt`
  - Truncation command: `.venv/bin/python scripts/evaluate_truncation.py --checkpoint checkpoints/cifar10_headline_mps_20260412-223650.pt --split test --device mps`
  - Top eigenvalue summary: class `1`, eigenvalue `0.010373540222644806`
  - Key truncation points:
    - rank `16` accuracy `0.3198`
    - rank `64` accuracy `0.4101`
    - rank `128` accuracy `0.4175`
    - full rank `512` accuracy `0.4181`
  - Interpretation: longer training improves accuracy but does not make CIFAR-10 low-rank in the MNIST sense; the spectrum still matters broadly.

- CIFAR-10 width ablation, local `uv` environment, config `configs/cifar10_width1024_mps.yaml`
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/cifar10_width1024_mps.yaml`
  - Hardware: `mps`
  - Run name: `cifar10_width1024_mps_20260412-223946`
  - Epochs: `10`
  - Best val accuracy: `0.4062`
  - Test accuracy at best validation checkpoint: `0.3950`
  - Truncation summary:
    - rank `64` accuracy `0.3907`
    - rank `128` accuracy `0.3939`
    - full rank `1024` accuracy `0.3950`
  - Interpretation: doubling width alone does not materially improve the 10-epoch CIFAR result.

- CIFAR-10 lower-weight-decay ablation, local `uv` environment, config `configs/cifar10_wd001_mps.yaml`
  - Command: `.venv/bin/python scripts/train_baseline.py --config configs/cifar10_wd001_mps.yaml`
  - Hardware: `mps`
  - Run name: `cifar10_wd001_mps_20260412-224113`
  - Epochs: `10`
  - Best val accuracy: `0.3942`
  - Test accuracy at best validation checkpoint: `0.3974`
  - Truncation summary:
    - rank `64` accuracy `0.3869`
    - rank `128` accuracy `0.3960`
    - full rank `512` accuracy `0.3974`
  - Interpretation: lowering weight decay from `0.05` to `0.01` does not materially change the CIFAR result.

### Verification

- Base commit before Task D edits: `278626e21731bdf6321b71672f92a2b10c274593`
- Syntax verification: `python3 -m compileall src scripts`
- Docker image build verification: `docker build -t reading-the-weights .`
- Docker smoke test verification:
  `docker run --rm reading-the-weights python scripts/smoke_test.py --config configs/mnist_baseline.yaml`
- Smoke test result: passed
- Smoke test artifacts inside container:
  - `results/metrics/mnist_baseline_smoke_20260413-020742/metrics.csv`
  - `checkpoints/mnist_baseline_smoke_20260413-020742.pt`
  - `results/analysis/mnist_baseline_smoke_20260413-020742/decomposition.pt`
- Local CIFAR-10 smoke test result: passed
- Local CIFAR-10 pilot training result: passed after CPU-safe data-loader fallback fix
- Local CIFAR-10 MPS pilot training result: passed
- Local MNIST MPS pilot training result: passed
- Local truncation analysis for MNIST and CIFAR-10 checkpoints: passed
- Local CIFAR-10 headline MPS baseline result: passed
- Local CIFAR-10 width ablation result: passed
- Local CIFAR-10 lower-weight-decay ablation result: passed

### Notes

- The Docker build currently succeeds with the default Linux PyTorch resolution path, which pulls a large NVIDIA-enabled dependency stack.
- Before scaling up CIFAR-10 runs, revisit the container dependency path to decide whether a CPU-only PyTorch install is preferable for lighter reproducibility.
- The first unsandboxed local CIFAR smoke run failed until CIFAR-10 was downloaded with network access.
- The first local CPU pilot run failed with a macOS `torch_shm_manager` shared-memory permission error when `num_workers > 0`.
- The training pipeline now forces a safe single-process data-loading path on sandboxed macOS CPU runs.
- No checkpoint format, decomposition output structure, or model public property changes were made.
- `build_image_dataloaders` now returns `val_loader` in addition to `train_loader` and `test_loader`; this is the main shared API change to mention to teammates.
- The first Task D-aligned comparison is now in hand:
  - MNIST reaches near-full accuracy by rank `8-16`.
  - CIFAR-10 keeps improving far beyond rank `16` and is still materially dependent on rank `64-128+`.
  - This supports the claim that CIFAR-10 is not just a harder MNIST case for this method.
- First ablation read:
  - More training helps CIFAR accuracy modestly.
  - Width alone does not rescue the method.
  - Lower weight decay does not rescue the method.
  - The next serious move should be a locality-preserving redesign rather than more scalar hyperparameter tuning.
