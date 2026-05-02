# Results Table

This file tracks the most informative local Task D runs.

| Run | Config | Device | Epochs | Best Val Acc | Test Acc | Key Truncation Result | Interpretation |
|---|---|---:|---:|---:|---:|---|---|
| `mnist_pilot_mps_20260412-223552` | `configs/mnist_pilot_mps.yaml` | `mps` | 10 | 0.9672 | 0.9684 | rank 8: 0.9640, rank 16: 0.9686, full: 0.9684 | MNIST is strongly low-rank under this method. |
| `cifar10_pilot_mps_20260412-221450` | `configs/cifar10_pilot_mps.yaml` | `mps` | 10 | 0.3942 | 0.3967 | rank 64: 0.3824, rank 128: 0.3938, full: 0.3967 | CIFAR pilot already uses a broad spectrum. |
| `cifar10_headline_mps_20260412-223650` | `configs/cifar10_headline_mps.yaml` | `mps` | 25 | 0.4258 | 0.4181 | rank 64: 0.4101, rank 128: 0.4175, full: 0.4181 | Longer training improves accuracy, but CIFAR still does not compress like MNIST. |
| `cifar10_locality_mps_20260412-230328` | `configs/cifar10_locality_mps.yaml` | `mps` | 25 | 0.4328 | 0.4200 | rank 32: 0.4037, rank 64: 0.4215, rank 128: 0.4201, full: 0.4200 | Locality-preserving preprocessing modestly steepens the truncation curve at matched budget. |
| `cifar10_locality4_mps_20260412-231256` | `configs/cifar10_locality4_mps.yaml` | `mps` | 25 | 0.4326 | 0.4399 | rank 16: 0.3924, rank 32: 0.4290, rank 64: 0.4402, full: 0.4399 | Stronger locality bias is the best redesign result so far: much steeper truncation and nearly the same accuracy as the 50-epoch raw-pixel baseline. |
| `cifar10_locality4_seed123_mps_20260412-231948` | `configs/cifar10_locality4_seed123_mps.yaml` | `mps` | 25 | 0.4360 | 0.4436 | rank 16: 0.3936, rank 32: 0.4324, rank 64: 0.4434, full: 0.4436 | Replication confirms the `4x4` locality result is stable across seeds and slightly better than seed 42. |
| `cifar10_locality4_50e_mps_20260427-224921` | `configs/cifar10_locality4_50e_mps.yaml` | `mps` | 50 | 0.4574 | 0.4612 | rank 32: 0.4430, rank 64: 0.4594, full: 0.4612 | Longer `4x4` training clears Kai's raw-student Task G baseline while preserving strong rank efficiency. |
| `cifar10_locality4_wd001_50e_mps_20260427-225245` | `configs/cifar10_locality4_wd001_50e_mps.yaml` | `mps` | 50 | 0.4592 | 0.4638 | rank 32: 0.4408, rank 64: 0.4630, full: 0.4638 | Current best pooling result: small gain from lower weight decay and near-full performance by rank 64. |
| `cifar10_locality2_50e_mps_20260427-225510` | `configs/cifar10_locality2_50e_mps.yaml` | `mps` | 50 | 0.4566 | 0.4509 | rank 32: 0.4049, rank 64: 0.4451, full: 0.4509 | `2x2` pooling is competitive on validation but less accurate on test and less rank-efficient than `4x4`. |
| `cifar10_width1024_mps_20260412-223946` | `configs/cifar10_width1024_mps.yaml` | `mps` | 10 | 0.4062 | 0.3950 | rank 64: 0.3907, rank 128: 0.3939, full: 0.3950 | Width alone does not meaningfully improve the 10-epoch CIFAR result. |
| `cifar10_wd001_mps_20260412-224113` | `configs/cifar10_wd001_mps.yaml` | `mps` | 10 | 0.3942 | 0.3974 | rank 64: 0.3869, rank 128: 0.3960, full: 0.3974 | Lower weight decay does not materially change the story. |
| `cifar10_completion_mps_20260412-225712` | `configs/cifar10_completion_mps.yaml` | `mps` | 50 | 0.4452 | 0.4408 | rank 64: 0.4280, rank 128: 0.4424, full: 0.4408 | Raw-pixel CIFAR improves with longer training but still remains broad-spectrum. |

## Current best read

- The MNIST/CIFAR truncation contrast is now robust enough to treat as a real finding, not a logging artifact.
- The current raw-pixel CIFAR setup improves with more training, but not in a way that makes the spectrum sharply low-rank.
- Scalar controls, including width and lower weight decay, do not explain the redesign gain.
- A stronger `4x4` locality-preserving redesign improves both the spectral story and training efficiency relative to raw-pixel CIFAR.
- Replication confirms the `4x4` locality effect is stable across at least two seeds.
- Longer/tuned `4x4` pooling now clears Kai's raw-student Task G validation baseline and reaches `46.38%` test accuracy.
- The next productive move is packaging the raw-vs-pooled comparison cleanly for the report.
