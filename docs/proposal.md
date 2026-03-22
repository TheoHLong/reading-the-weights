# CS 7643 Deep Learning — Project Proposal

**Team Name:** ？？？

**Project Title:** Reading the Weights: Interpretable Feature Extraction from Bilinear MLPs

------

## Project Summary

A central question in deep learning is what knowledge a trained network actually encodes, and where that knowledge lives. Current interpretability methods overwhelmingly answer this by probing *activations* on a dataset — but this only tells us which features are present for a given input, not how the weights construct those features or what they would do on unseen inputs. Ideally, we could read a model's parameters the way we read source code: understanding its behavior by inspecting the weights alone. Pearce et al. (2025) opened this door by showing that bilinear MLPs — a GLU variant that removes the element-wise nonlinearity, computing g(x) = (Wx) ⊙ (Vx) — can be fully expressed as a third-order tensor and decomposed via eigendecomposition into interpretable features directly from the weights, with no input data required. Their top eigenvectors correspond to visually meaningful structures (e.g., digit-specific stroke detectors on MNIST) and exhibit low-rank spectra, suggesting that bilinear layers learn surprisingly compact representations. However, two questions remain open: does this interpretable structure survive on natural images beyond toy datasets, and can we use this weight-level lens to observe phenomena that activation-based methods cannot — such as what happens to a network's internal feature vocabulary when architectural inductive biases are transferred from another network? Our project replicates the core bilinear MLP results and pursues both questions, connecting weight-based interpretability to the study of inductive bias transfer.

------

## Approach

**Core replication (Bilinear MLP on MNIST/FMNIST).** We implement a custom bilinear layer in PyTorch from scratch (forward pass, bilinear tensor construction, symmetrization), train shallow bilinear classifiers on MNIST and Fashion-MNIST, and build an eigendecomposition pipeline to extract and visualize interpretable eigenvectors per output class. We replicate four key results from the paper: top eigenvector visualization (their Fig. 2–3), eigenvalue truncation vs. accuracy (their Fig. 5B), the effect of Gaussian input noise on eigenvector quality (their Fig. 4), and adversarial mask construction via pseudoinverse (their Fig. 7). We use the authors' open-source code (https://github.com/tdooms/bilinear-decomposition) as a reference for verification, but re-implement the core analysis pipeline ourselves.

**Extension 1: CIFAR-10.** The original paper only tests on grayscale toy datasets. We extend to CIFAR-10 to examine whether eigenvectors remain interpretable on natural color images and how the eigenvalue rank compares — this directly tests the generality of the approach.

**Extension 2 (stretch goal): Inductive bias transfer + eigenvector analysis.** We investigate how transferring inductive bias from a guide network (e.g., ResNet-18) reshapes the weight structure of a bilinear model, by comparing eigenvector spectra before vs. after the transfer. The transfer method will be chosen based on our available compute: our primary approach is CKA-based guidance from "Training the Untrainable" (Subramaniam et al., 2025), which aligns layerwise representations during training but requires storing activation tensors for both networks at every guided layer, making it memory-intensive. If GPU memory proves insufficient, we fall back to knowledge distillation (also evaluated in the same paper, their Appendix H), which only requires matching output logits and is far cheaper. Either method lets us ask the same core question: can we see, in the weights, what inductive bias transfer looks like? We measure this via effective rank changes, qualitative eigenvector comparisons, and validation loss curves. This stage is treated as a stretch goal — if neither method converges within our timeline, the core replication plus CIFAR-10 extension already constitute a complete project.

------

## Resources / Related Work & Papers

The primary paper we replicate is Pearce et al. (2025), "Bilinear MLPs Enable Weight-Based Mechanistic Interpretability" (ICLR 2025), which introduced the eigendecomposition framework for bilinear layers and demonstrated low-rank interpretable structure on image classification and language modeling tasks. The theoretical motivation comes from Sharkey (2023), who first proposed that bilinear layers are intrinsically interpretable because their computations can be expressed as linear operations on a third-order tensor. For our extension, we build on Subramaniam et al. (2025), "Training the Untrainable" (NeurIPS 2025), which showed that even randomly initialized guide networks can transfer useful architectural priors via CKA-based representational alignment. CKA itself was introduced by Kornblith et al. (2019) as a metric for comparing neural network representations and has since become standard in representation analysis. The broader context for this work is the mechanistic interpretability literature, where activation-based methods like sparse autoencoders (Cunningham et al., 2024; ICLR 2024) dominate — our project explores the complementary weight-based direction.

### References

- **Pearce, M. T., Dooms, T., Oramas, J., Rigg, A., & Sharkey, L. (2025).** Bilinear MLPs enable weight-based mechanistic interpretability. *To appear in International Conference on Learning Representations (ICLR)*.
- **Subramaniam, V., Mayo, D., Conwell, C., Poggio, T., Katz, B., Cheung, B., & Barbu, A. (2024).** Training the Untrainable: Introducing Inductive Bias via Representational Alignment. *Advances in Neural Information Processing Systems (NeurIPS)*.
- **Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).** Similarity of neural network representations revisited. *International Conference on Machine Learning (ICML)*.
- **Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2024).** Sparse Autoencoders Find Highly Interpretable Features in Language Models. *International Conference on Learning Representations (ICLR)*.
- **Sharkey, L. (2023).** Bilinear Layers for Interpretability. *(Alignment Forum / AI Safety Technical Report)*.

------

## Datasets

- **MNIST**: http://yann.lecun.com/exdb/mnist/ — 70,000 grayscale 28×28 digit images, 10 classes.
- **Fashion-MNIST**: https://github.com/zalandoresearch/fashion-mnist — 70,000 grayscale 28×28 clothing images, 10 classes.
- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html — 60,000 color 32×32 natural images, 10 classes.

All datasets are publicly available via torchvision with standard train/test splits. No custom data collection or annotation is needed.

------

## Group Members

- Tenghai Long
- Akshayalakshmi Venkattakuppan Krishnakumar
- Aliya Abdul Rahim
