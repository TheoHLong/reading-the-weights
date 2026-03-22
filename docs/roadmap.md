# Roadmap

## Milestone 0: Repository setup
- keep upstream reference code separate
- track configs, scripts, notes, and small summaries in git
- store large artifacts in Drive or local ignored folders

## Milestone 1: MNIST baseline
- train a single-layer bilinear MLP
- reproduce eigendecomposition plots
- log validation accuracy and truncation experiments

## Milestone 2: Fashion-MNIST replication
- verify the same pipeline on a second grayscale dataset
- compare spectra and qualitative eigenvectors

## Milestone 3: CIFAR-10 extension
- add color-image dataset support
- update plotting and decomposition utilities for 32x32x3 inputs

## Milestone 4: Stretch goal
- only after the baseline is stable
- prefer distillation before CKA on Colab due to memory pressure
