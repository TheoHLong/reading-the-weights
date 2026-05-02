# Task C Narrative

Task C implements the visualization side of the bilinear-weight analysis. For each output class, we eigendecompose the class-specific symmetrized bilinear matrix, select the top positive eigendirection, project the hidden-space eigenvector back into input space through the learned embedding matrix, reshape it as an image, and plot the result.

The initial unregularized figures were too pixel-artifact-heavy for a serious report figure. The final Task C figures therefore use paper-aligned training-only Gaussian input noise to reduce outlying-pixel overfit while preserving the same checkpoint and decomposition interfaces.

## Report Figures

- `mnist_figure2_style_eigenvectors.png`: recommended main MNIST panel, matching the paper's compact Figure-2-style layout.
- `fmnist_figure2_style_eigenvectors.png`: recommended main Fashion-MNIST panel, using representative classes `trouser`, `pullover`, `dress`, `coat`, and `sandal`.
- `task_c_eigenvector_panel.png`: supplemental all-class panel.
- `mnist_top_eigenvectors.png`: supplemental all-class MNIST grid.
- `fmnist_regularized_all_eigenvectors.png`: supplemental all-class Fashion-MNIST grid.

## Runs

- MNIST visualization checkpoint: `mnist_task_c_noise040_mps_20260501-215725`, best validation accuracy `98.22%`, test accuracy `98.16%`.
- Fashion-MNIST visualization checkpoint: `fmnist_task_c_noise040_mps_20260501-215229`, best validation accuracy `87.78%`, test accuracy `87.13%`.

These are visualization-quality checkpoints. Use the Task A / baseline numbers when discussing best MNIST or Fashion-MNIST classifier accuracy.

## Suggested Caption

Representative top input-space eigenvectors for regularized single-layer bilinear classifiers on MNIST and Fashion-MNIST. Hidden-space eigendirections are projected through the learned input embedding and reshaped as images. MNIST eigenvectors recover digit-stroke templates, while Fashion-MNIST eigenvectors behave more like clothing-edge templates. Each panel is independently scaled; color indicates signed eigenvector structure, not positive/negative class evidence by itself.
