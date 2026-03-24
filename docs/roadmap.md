# Task A Roadmap

## Boundary
- stay within MNIST and Fashion-MNIST
- stay within single-layer bilinear baseline models
- produce stable artifacts that unblock later analysis and visualization work

## Milestone A1: Core reimplementation
- implement bilinear layer forward pass
- implement class-wise bilinear tensor construction
- implement tensor symmetrization
- implement eigendecomposition and projection back to input space

## Milestone A2: Training framework
- dataloader for MNIST and Fashion-MNIST
- training loop, optimizer, scheduler
- checkpoint saving for best and latest models
- per-epoch metrics export

## Milestone A3: Baseline runs
- train MNIST baseline
- train Fashion-MNIST baseline
- verify checkpoints can be decomposed without notebook-only code

## Milestone A4: Handoff contract
- hand off checkpoints, metrics, and decomposition artifacts
- keep visualization and figure replication out of Task A code
