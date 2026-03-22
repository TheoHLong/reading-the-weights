# Reading the Weights

Clean experiment workspace for the CS7643 final project on bilinear MLP weight interpretability.

## Why this folder exists

This repository is the working area for your own experiments. The official reference implementation stays separate in `../bilinear-decomposition`, so your code, configs, notes, and results do not get mixed with upstream code.

## Layout

- `src/reading_weights/`: your project package
- `scripts/`: script-first training and analysis entry points
- `configs/`: experiment configs tracked in git
- `docs/`: proposal snapshot and working notes
- `notebooks/`: light notebooks for Colab demos and visualization only
- `results/`: small tracked result summaries; large outputs should stay out of git
- `checkpoints/`: local or Drive-backed model checkpoints
- `data/`: local dataset cache only

## Colab workflow

1. Push this folder to GitHub.
2. Open Colab and mount Google Drive.
3. Clone the repo into the Colab runtime.
4. Run `bash scripts/setup_colab.sh`.
5. Save checkpoints and large result artifacts to Drive, not to git.

## Current baseline

- `scripts/train_mnist.py`: trains the first MNIST baseline from `configs/mnist_baseline.yaml`
- `scripts/analyze_mnist.py`: saves eigendecomposition artifacts for a trained checkpoint
- `notebooks/colab_quickstart.ipynb`: thin Colab entrypoint for setup and script execution

## Immediate next steps

1. Connect this repo to GitHub and push the initial scaffold.
2. Run the MNIST baseline in Colab and verify the first checkpoint and metrics export.
3. Improve the analysis plots to match the paper figures before extending to Fashion-MNIST.
4. Keep the transfer-learning idea as a later milestone.
