# Task D Narrative

## Main finding

The CIFAR extension now supports a clean compatibility story:

1. MNIST is strongly low-rank under the bilinear decomposition framework.
2. Raw-pixel CIFAR-10 trains, but remains broad-spectrum even after longer training.
3. Scalar controls such as width, weight decay, and tiny L1 do not fix that broad-spectrum behavior.
4. Adding locality-preserving preprocessing makes CIFAR-10 substantially more compressible under the same decomposition analysis.
5. The benefit is not monotone: `4x4` pooling is a strong regime, while `16x16` pooling overcompresses the task.

## Raw-pixel completion

- Best raw-pixel completion run: `cifar10_completion_mps_20260412-225712`
- Best validation accuracy: `0.4452`
- Test accuracy: `0.4408`
- Rank-64 accuracy: `0.4280`
- Full-rank accuracy: `0.4408`

Interpretation: the raw-pixel CIFAR extension is complete, but the resulting spectrum remains broad.

## Failed controls

- Matched-budget tiny-L1 control: `cifar10_l1tiny_mps_20260422-203929`, test accuracy `0.4180`, rank-64 accuracy `0.4100`

Interpretation: scalar regularization does not explain the pooled-model gain.

## Locality-preserving redesign

- Seed 42 run: `cifar10_locality4_mps_20260412-231256`, test accuracy `0.4399`, rank-64 accuracy `0.4402`
- Seed 123 run: `cifar10_locality4_seed123_mps_20260412-231948`, test accuracy `0.4436`, rank-64 accuracy `0.4434`
- Tuned 50-epoch run: `cifar10_locality4_wd001_50e_mps_20260427-225245`, test accuracy `0.4638`, rank-64 accuracy `0.4630`

Interpretation: the `4x4` locality-preserving redesign is stable across two seeds, clears the raw-student Task G baseline when trained for 50 epochs, and reaches essentially full-rank performance by rank 64.

## Locality-strength tradeoff

- Intermediate `8x8` pooling: `cifar10_locality8_mps_20260422-204446`, test accuracy `0.3812`, rank-32 accuracy `0.3805`
- Aggressive `16x16` pooling: `cifar10_locality16_mps_20260422-204115`, test accuracy `0.2556`, rank-8 accuracy `0.2493`
- Replication: `cifar10_locality16_seed123_mps_20260422-204255`, test accuracy `0.2570`, rank-8 accuracy `0.2543`

Interpretation: stronger pooling keeps making the spectrum more compressible, but beyond `4x4` it removes too much semantic content from CIFAR-10.

## Report recommendation

Frame Extension 1 / Task I as a compatibility result:

- Raw-pixel CIFAR-10 shows that the MNIST result does not directly transfer.
- Scalar controls do not fix the problem.
- Locality-preserving CIFAR-10 shows that the weight-based analysis becomes much more compatible with natural images once basic spatial structure is restored.
- The best regime is moderate locality bias, not maximal pooling.
