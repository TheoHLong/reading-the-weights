# Results Table

This file tracks the most informative local Task D runs.

| Run | Config | Device | Epochs | Best Val Acc | Test Acc | Key Truncation Result | Interpretation |
|---|---|---:|---:|---:|---:|---|---|
| `mnist_pilot_mps_20260412-223552` | `configs/mnist_pilot_mps.yaml` | `mps` | 10 | 0.9672 | 0.9684 | rank 8: 0.9640, rank 16: 0.9686, full: 0.9684 | MNIST is strongly low-rank under this method. |
| `cifar10_pilot_mps_20260412-221450` | `configs/cifar10_pilot_mps.yaml` | `mps` | 10 | 0.3942 | 0.3967 | rank 64: 0.3824, rank 128: 0.3938, full: 0.3967 | CIFAR pilot already uses a broad spectrum. |
| `cifar10_headline_mps_20260412-223650` | `configs/cifar10_headline_mps.yaml` | `mps` | 25 | 0.4258 | 0.4181 | rank 64: 0.4101, rank 128: 0.4175, full: 0.4181 | Longer training improves accuracy, but CIFAR still does not compress like MNIST. |
| `cifar10_locality_mps_20260412-230328` | `configs/cifar10_locality_mps.yaml` | `mps` | 25 | 0.4328 | 0.4200 | rank 32: 0.4037, rank 64: 0.4215, rank 128: 0.4201, full: 0.4200 | Locality-preserving preprocessing modestly steepens the truncation curve at matched budget. |
| `cifar10_width1024_mps_20260412-223946` | `configs/cifar10_width1024_mps.yaml` | `mps` | 10 | 0.4062 | 0.3950 | rank 64: 0.3907, rank 128: 0.3939, full: 0.3950 | Width alone does not meaningfully improve the 10-epoch CIFAR result. |
| `cifar10_wd001_mps_20260412-224113` | `configs/cifar10_wd001_mps.yaml` | `mps` | 10 | 0.3942 | 0.3974 | rank 64: 0.3869, rank 128: 0.3960, full: 0.3974 | Lower weight decay does not materially change the story. |
| `cifar10_completion_mps_20260412-225712` | `configs/cifar10_completion_mps.yaml` | `mps` | 50 | 0.4452 | 0.4408 | rank 64: 0.4280, rank 128: 0.4424, full: 0.4408 | Raw-pixel CIFAR improves with longer training but still remains broad-spectrum. |

## Current best read

- The MNIST/CIFAR truncation contrast is now robust enough to treat as a real finding, not a logging artifact.
- The current raw-pixel CIFAR setup improves with more training, but not in a way that makes the spectrum sharply low-rank.
- The first locality-preserving redesign modestly improves spectral concentration at matched epoch budget.
- The next productive move is likely a stronger locality-preserving redesign or a replication run to confirm the effect.
