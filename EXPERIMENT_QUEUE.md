# Experiment Queue

This file tracks pending, running, and completed local experiments for the CIFAR compatibility study.

## Current best

- Best raw-pixel reference: `cifar10_completion_mps_20260412-225712`
- Config: `configs/cifar10_completion_mps.yaml`
- Best val accuracy: `0.4452`
- Test accuracy at best validation checkpoint: `0.4408`
- Best compatibility result: `cifar10_locality4_wd001_50e_mps_20260427-225245`
- Config: `configs/cifar10_locality4_wd001_50e_mps.yaml`
- Best val accuracy: `0.4592`
- Test accuracy at best validation checkpoint: `0.4638`

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
  - Status: lower-weight-decay controls completed, no clear improvement outside the tuned `4x4` setting

- `locality-redesign`
  - Goal: test whether fixed locality-preserving preprocessing steepens the CIFAR truncation curve
  - Status: report-facing comparison uses `2x2` and `4x4`; `4x4` is the current sweet spot

## Deferred until baseline quality improves

- Any new model family beyond the fixed locality-preprocessing line
- Additional scalar regularization sweeps
- Any report-quality qualitative interpretation claims that are not backed by replicated runs

## Current interpretation

- Longer raw-pixel CIFAR training improves accuracy modestly but does not create MNIST-like spectral concentration.
- MNIST retains near-full performance by rank `8-16`, while CIFAR-10 still benefits materially through rank `64-128+`.
- Width and lower weight decay do not materially change the conclusion.
- Locality-preserving preprocessing creates a real compatibility regime rather than a simple accuracy hack.
- `4x4` pooling gives the best redesign result so far, with rank-32/64 performance effectively at full-rank accuracy across two seeds and after 50-epoch calibration.
- The current paper-facing claim should emphasize the raw-vs-pooled contrast without going deep into secondary controls.
