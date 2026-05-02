# Task C Worklog

This file records the eigenvector visualization work for Task C: build a visualization module and replicate the Fig. 2/3-style top input-space eigenvector analysis.

Integration note: the original commands below used flat `configs/*.yaml` paths. In the integrated branch, these Task C visualization configs live under `configs/task_c/`.

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
- `configs/mnist_task_c_noise040_mps.yaml`
- `configs/fmnist_task_c_noise015_mps.yaml`
- `configs/fmnist_task_c_noise030_mps.yaml`
- `configs/fmnist_task_c_noise035_mps.yaml`
- `configs/fmnist_task_c_noise040_mps.yaml`
- `configs/fmnist_task_c_h512_noise030_mps.yaml`
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

### Figure critique and final report asset revision

After comparing the generated figures against the paper's Figure 2 style, the all-class Fashion-MNIST grid is still too visually uneven for a main report figure. This is not a plotting bug: it is the same overfitting/artifact failure mode discussed by the paper/tutorial for weakly regularized image classifiers.

Additional local sweep:

- `fmnist_task_c_noise015_mps_20260501-214608`: `train_input_noise_std=0.15`, best validation `89.28%`, test `88.40%`; figure remained too artifact-heavy.
- `fmnist_task_c_noise030_mps_20260501-214753`: `train_input_noise_std=0.30`, best validation `88.68%`, test `87.84%`; clearer but still uneven.
- `fmnist_task_c_h512_noise030_mps_20260501-214953`: `d_hidden=512`, `train_input_noise_std=0.30`, best validation `89.02%`, test `88.18%`; width did not materially solve the artifact problem.
- `fmnist_task_c_noise035_mps_20260501-215439`: `train_input_noise_std=0.35`, best validation `88.28%`, test `87.53%`; no clear advantage over `0.40`.
- `fmnist_task_c_noise040_mps_20260501-215229`: `train_input_noise_std=0.40`, best validation `87.78%`, test `87.13%`; cleanest representative Fashion-MNIST eigenvectors.
- `mnist_task_c_noise040_mps_20260501-215725`: `train_input_noise_std=0.40`, best validation `98.22%`, test `98.16%`; strong MNIST visualization checkpoint.

Final report-facing Task C figures:

- `report_assets/task_c/mnist_figure2_style_eigenvectors.png`
  - Source: `results/analysis/mnist_task_c_noise040_mps_20260501-215725/mnist_task_c_noise040_mps_20260501-215725/decomposition.pt`
  - Classes shown: MNIST `0,1,2,3,4`
  - Command: `.venv/bin/python scripts/visualize_eigenvectors.py --decomposition results/analysis/mnist_task_c_noise040_mps_20260501-215725/mnist_task_c_noise040_mps_20260501-215725/decomposition.pt --output report_assets/task_c/mnist_figure2_style_eigenvectors.png --dataset mnist --image-size 28 --channels 1 --class-indices 0,1,2,3,4 --number-panels`
- `report_assets/task_c/fmnist_figure2_style_eigenvectors.png`
  - Source: `results/analysis/fmnist_task_c_noise040_mps_20260501-215229/fmnist_task_c_noise040_mps_20260501-215229/decomposition.pt`
  - Classes shown: Fashion-MNIST `trouser,pullover,dress,coat,sandal`
  - Command: `.venv/bin/python scripts/visualize_eigenvectors.py --decomposition results/analysis/fmnist_task_c_noise040_mps_20260501-215229/fmnist_task_c_noise040_mps_20260501-215229/decomposition.pt --output report_assets/task_c/fmnist_figure2_style_eigenvectors.png --dataset fashion_mnist --image-size 28 --channels 1 --class-indices 1,2,3,4,5 --number-panels`
- `report_assets/task_c/fmnist_regularized_all_eigenvectors.png`
  - Supplemental all-class Fashion-MNIST grid; useful for transparency but not recommended as a main paper figure.

### Current read

- Task C is now implemented in the reproducible and report-quality sense.
- The report should use the Figure-2-style selected panels as the main Task C qualitative figure.
- The all-class grids are useful supplementary assets but should not be the main Fashion-MNIST figure.
- The regularized Task C checkpoints are intended for qualitative eigenvector clarity; they should not replace the higher-accuracy baseline numbers when discussing classifier performance.
