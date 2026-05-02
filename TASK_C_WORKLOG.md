# Task C Worklog

This file records the eigenvector visualization work for Task C: build a visualization module and replicate the Fig. 2/3-style top input-space eigenvector analysis.

## 2026-05-01

### Initial implementation

- Added `scripts/visualize_eigenvectors.py`.
- The script reads a saved `decomposition.pt`, selects the top input-space eigenvector for each output class, reshapes it to image space, and writes a class-grid PNG.
- Default report behavior uses:
  - top positive eigenvalue per class
  - sign orientation by largest absolute entry
  - per-image robust color scaling
  - class labels without eigenvalue clutter

### Figure-quality correction

The first Task C figures were mathematically correct but visually too noisy: the top projected eigenvectors emphasized outlying pixel artifacts rather than coherent digit/clothing structure. This matched the failure mode described in the source paper/tutorial for unregularized image classifiers.

To fix this without changing checkpoint or decomposition formats, training now supports optional training-only Gaussian input noise via `train.train_input_noise_std`. This is used only during optimization and is disabled for validation, test evaluation, checkpoint loading, and decomposition.

Code/config additions:

- `src/reading_weights/train.py`: optional `train_input_noise_std`
- `configs/mnist_task_c_noise_mps.yaml`
- `configs/fmnist_task_c_noise_mps.yaml`
- `scripts/build_task_c_report_assets.py`

### MNIST regularized visualization run

- Training command: `.venv/bin/python scripts/train_baseline.py --config configs/mnist_task_c_noise_mps.yaml`
- Run: `mnist_task_c_noise_mps_20260501-215026`
- Best validation accuracy: `0.9763`
- Test accuracy at best validation checkpoint: `0.9767`
- Decomposition command: `.venv/bin/python scripts/analyze_checkpoint.py --checkpoint checkpoints/mnist_task_c_noise_mps_20260501-215026.pt`
- All-class figure command: `.venv/bin/python scripts/visualize_eigenvectors.py --decomposition results/analysis/mnist_task_c_noise_mps_20260501-215026/decomposition.pt --output report_assets/task_c/mnist_top_eigenvectors.png --dataset mnist --image-size 28 --channels 1`
- Output: `report_assets/task_c/mnist_top_eigenvectors.png`

### Fashion-MNIST regularized visualization run

- Training command: `.venv/bin/python scripts/train_baseline.py --config configs/fmnist_task_c_noise_mps.yaml`
- Run: `fmnist_task_c_noise_mps_20260501-214848`
- Best validation accuracy: `0.8695`
- Test accuracy at best validation checkpoint: `0.8580`
- Decomposition command: `.venv/bin/python scripts/analyze_checkpoint.py --checkpoint checkpoints/fmnist_task_c_noise_mps_20260501-214848.pt`
- All-class figure command: `.venv/bin/python scripts/visualize_eigenvectors.py --decomposition results/analysis/fmnist_task_c_noise_mps_20260501-214848/decomposition.pt --output report_assets/task_c/fmnist_top_eigenvectors.png --dataset fashion_mnist --image-size 28 --channels 1`
- Output: `report_assets/task_c/fmnist_top_eigenvectors.png`

### Main report panel

- Command: `.venv/bin/python scripts/build_task_c_report_assets.py --mnist-decomposition results/analysis/mnist_task_c_noise_mps_20260501-215026/decomposition.pt --fmnist-decomposition results/analysis/fmnist_task_c_noise_mps_20260501-214848/decomposition.pt --output report_assets/task_c/task_c_eigenvector_panel.png`
- Output: `report_assets/task_c/task_c_eigenvector_panel.png`
- Contents: all 10 MNIST classes and all 10 Fashion-MNIST classes, arranged as a readable `4 x 5` report panel.
- Layout: first two rows are MNIST digits `0-9`; last two rows are Fashion-MNIST classes `t-shirt`, `trouser`, `pullover`, `dress`, `coat`, `sandal`, `shirt`, `sneaker`, `bag`, `ankle boot`.

### Current read

- Task C is now implemented in the reproducible and report-quality sense.
- The report should use the all-class `task_c_eigenvector_panel.png` as the main Task C figure.
- The all-class grids are useful supplementary assets.
- The regularized Task C checkpoints are intended for qualitative eigenvector clarity; they should not replace the higher-accuracy baseline numbers when discussing classifier performance.
