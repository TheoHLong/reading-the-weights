# Experiment Queue

This file tracks pending, running, and completed local experiments for Task D.

## Current best

- Best logged CIFAR baseline so far: `cifar10_headline_mps_20260412-223650`
- Config: `configs/cifar10_headline_mps.yaml`
- Best val accuracy: `0.4258`
- Test accuracy at best validation checkpoint: `0.4181`

## Next runs

- `width-ablation`
  - Goal: test whether wider hidden states improve CIFAR performance and spectral concentration
  - Config direction: compare `d_hidden` values against `configs/cifar10_headline_mps.yaml`
  - Status: first width-1024 pilot completed, no clear improvement

- `truncation-eval`
  - Goal: reproduce Task D truncation-vs-accuracy analysis on MNIST and CIFAR-10
  - Status: completed for current MNIST and CIFAR MPS checkpoints

- `regularization-ablation`
  - Goal: test whether stronger or modified regularization improves CIFAR spectra
  - Status: first lower-weight-decay pilot completed, no clear improvement

- `locality-redesign`
  - Goal: test whether fixed locality-preserving preprocessing steepens the CIFAR truncation curve
  - Status: first `2x2` redesign promising; stronger `4x4` redesign currently strongest result

## Deferred until baseline quality improves

- Patchified/locality-preserving redesign
- L1-based sparse bilinear experiments
- Any report-quality qualitative interpretation claims

## Current interpretation

- Longer raw-pixel CIFAR training improves accuracy modestly but does not create MNIST-like spectral concentration.
- MNIST retains near-full performance by rank `8-16`, while CIFAR-10 still benefits materially through rank `64-128+`.
- First width and weight-decay ablations did not materially change the conclusion.
- First locality-preserving redesign slightly steepens the truncation curve at matched 25-epoch budget.
- Stronger `4x4` pooling gives the best redesign result so far, with rank-32/64 performance much closer to full-rank accuracy.
- The next iteration should prioritize replication and careful refinement of locality-preserving variants over more scalar hyperparameter tuning.
