# Configs

- `baselines/`: standalone bilinear student baselines and noise-regularized baseline configs.
- `guides/`: guide/teacher model training configs.
- `transfer/`: KD and CKA student transfer configs, including ablations and debug variants.

All scripts accept `--config`, so configs can be moved within this tree without changing experiment code.

## SQS Configs

SQS configs use `model.gate: sqs`, which maps to `SignedQuadraticShrink` in `src.model`.

- `mnist_paper_bilinear.yaml`, `mnist_paper_sqs.yaml`, `fmnist_paper_bilinear.yaml`, and `fmnist_paper_sqs.yaml` follow the SQS paper setup: `input_noise_std: 1.0`, `wd: 0.1`, `batch_size: 2048`, and `epochs: 20`.
- `cifar10_baseline_n1_sqs.yaml` and `cifar10_baseline_n4_sqs.yaml` are CIFAR-10 stability/performance ablations.
- `cifar10_cka_n1_sqs.yaml` and `cifar10_cka_n4_sqs.yaml` test whether CKA adds gains on top of SQS.

Use the paper-style MNIST/Fashion-MNIST configs for eigenspectrum comparisons such as `scripts/compare_eigenvectors.py`. The older `mnist_baseline*.yaml` and `fmnist_baseline*.yaml` configs are kept for the original course-project replication and do not match the SQS paper hyperparameters.
