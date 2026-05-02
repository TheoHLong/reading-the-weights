# Task C Narrative

Task C implements the visualization side of the bilinear-weight analysis. For each output class, we eigendecompose the class-specific symmetrized bilinear matrix and project the selected hidden-space eigenvector back into input space through the learned embedding matrix. The resulting vector is reshaped as an image and plotted in a class grid.

## Report Figures

- `mnist_top_eigenvectors.png`: top input-space eigenvector per MNIST output class.
- `fmnist_top_eigenvectors.png`: top input-space eigenvector per Fashion-MNIST output class.

## Runs

- MNIST: `mnist_task_c_mps_20260501-210901`, best validation accuracy `97.68%`, test accuracy `97.50%`.
- Fashion-MNIST: `fmnist_task_c_mps_20260501-210659`, best validation accuracy `89.10%`, test accuracy `88.06%`.

## Suggested Caption

Top input-space eigenvectors for each output class after projecting hidden-space eigendirections through the learned input embedding. Each panel is scaled independently to emphasize the spatial pattern of the selected class direction; color indicates signed contribution.
