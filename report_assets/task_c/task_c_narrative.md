# Task C Narrative

Task C implements the visualization side of the bilinear-weight analysis. For each output class, we eigendecompose the class-specific symmetrized bilinear matrix, select the top positive eigendirection, project the hidden-space eigenvector back into input space through the learned embedding matrix, reshape it as an image, and plot the result.

The initial unregularized figures were too pixel-artifact-heavy for a serious report figure. The final Task C figures therefore use paper-aligned training-only Gaussian input noise to reduce outlying-pixel overfit while preserving the same checkpoint and decomposition interfaces.

## Report Figures

- `task_c_eigenvector_panel.png`: main all-class report panel; first two rows show MNIST digits `0-9`, and last two rows show all Fashion-MNIST classes.
- `mnist_top_eigenvectors.png`: supplementary all-class MNIST grid.
- `fmnist_top_eigenvectors.png`: supplementary all-class Fashion-MNIST grid.

## Runs

- MNIST visualization checkpoint: `mnist_task_c_noise_mps_20260501-215026`, best validation accuracy `97.63%`, test accuracy `97.67%`.
- Fashion-MNIST visualization checkpoint: `fmnist_task_c_noise_mps_20260501-214848`, best validation accuracy `86.95%`, test accuracy `85.80%`.

These are visualization-quality checkpoints. Use the Task A / baseline numbers when discussing best MNIST or Fashion-MNIST classifier accuracy.

## Suggested Caption

Top input-space eigenvectors for a regularized single-layer bilinear classifier across all 10 MNIST classes and all 10 Fashion-MNIST classes. Hidden-space eigendirections are projected through the learned input embedding and reshaped as images. MNIST eigenvectors recover digit strokes, while Fashion-MNIST eigenvectors behave more like class-specific edge templates. Each panel is independently scaled; color indicates signed eigenvector structure, not positive/negative class evidence by itself.
