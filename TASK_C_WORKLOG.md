# Task C Worklog

This file records the eigenvector visualization work for Task C: build a visualization module and replicate the Fig. 2/3-style top input-space eigenvector per output class.

## 2026-05-01

### Code changes

- Added `scripts/visualize_eigenvectors.py`.
- The script reads a saved `decomposition.pt`, selects the top input-space eigenvector for each output class, reshapes it to image space, and writes a class-grid PNG.
- Default report behavior uses:
  - top positive eigenvalue per class
  - sign orientation by largest absolute entry
  - per-image robust color scaling
  - class labels without eigenvalue clutter

### MNIST figure

- Training command: `.venv/bin/python scripts/train_baseline.py --config configs/mnist_task_c_mps.yaml`
- Run: `mnist_task_c_mps_20260501-210901`
- Best validation accuracy: `0.9768`
- Test accuracy at best validation checkpoint: `0.9750`
- Decomposition command: `.venv/bin/python scripts/analyze_checkpoint.py --checkpoint checkpoints/mnist_task_c_mps_20260501-210901.pt`
- Figure command: `.venv/bin/python scripts/visualize_eigenvectors.py --decomposition results/analysis/mnist_task_c_mps_20260501-210901/decomposition.pt --output report_assets/task_c/mnist_top_eigenvectors.png --dataset mnist --image-size 28 --channels 1 --title 'MNIST Top Input-Space Eigenvector per Class'`
- Output: `report_assets/task_c/mnist_top_eigenvectors.png`

### Fashion-MNIST figure

- Training command: `.venv/bin/python scripts/train_baseline.py --config configs/fmnist_task_c_mps.yaml`
- Run: `fmnist_task_c_mps_20260501-210659`
- Best validation accuracy: `0.8910`
- Test accuracy at best validation checkpoint: `0.8806`
- Decomposition command: `.venv/bin/python scripts/analyze_checkpoint.py --checkpoint checkpoints/fmnist_task_c_mps_20260501-210659.pt`
- Figure command: `.venv/bin/python scripts/visualize_eigenvectors.py --decomposition results/analysis/fmnist_task_c_mps_20260501-210659/decomposition.pt --output report_assets/task_c/fmnist_top_eigenvectors.png --dataset fashion_mnist --image-size 28 --channels 1 --title 'Fashion-MNIST Top Input-Space Eigenvector per Class'`
- Output: `report_assets/task_c/fmnist_top_eigenvectors.png`

### Current read

- Task C is now implemented in the reproducible sense: a script can regenerate the class-wise top-eigenvector grids from decomposition artifacts.
- The report should frame these as qualitative replications of the Fig. 2/3 visualization procedure rather than as new quantitative claims.
