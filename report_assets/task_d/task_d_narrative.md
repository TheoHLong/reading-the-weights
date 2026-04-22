# Task D Narrative

## Main finding

Task D now supports a clean three-stage story:

1. MNIST is strongly low-rank under the bilinear decomposition framework.
2. Raw-pixel CIFAR-10 trains, but remains broad-spectrum even after longer training.
3. Adding locality-preserving preprocessing makes CIFAR-10 substantially more compressible under the same decomposition analysis.

## Raw-pixel completion

- Best raw-pixel completion run: `cifar10_completion_mps_20260412-225712`
- Best validation accuracy: `0.4452`
- Test accuracy: `0.4408`
- Rank-64 accuracy: `0.4280`
- Full-rank accuracy: `0.4408`

Interpretation: the raw-pixel CIFAR extension is complete, but the resulting spectrum remains broad.

## Locality-preserving redesign

- Seed 42 run: `cifar10_locality4_mps_20260412-231256`, test accuracy `0.4399`, rank-64 accuracy `0.4402`
- Seed 123 run: `cifar10_locality4_seed123_mps_20260412-231948`, test accuracy `0.4436`, rank-64 accuracy `0.4434`

Interpretation: the `4x4` locality-preserving redesign is stable across two seeds and reaches essentially full-rank performance by rank 64.

## Report recommendation

Frame Extension 1 as a negative-to-positive result:

- Raw-pixel CIFAR-10 shows that the MNIST result does not directly transfer.
- Locality-preserving CIFAR-10 shows that the weight-based analysis becomes much more compatible with natural images once basic spatial structure is restored.
