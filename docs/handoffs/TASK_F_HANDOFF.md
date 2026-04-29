# Task F Handoff

## Scope

Reimplementation of Pearce et al. (2025) Figure 7 for single-layer bilinear
image classifiers:

- decompose the checkpoint into input-space eigenvectors
- build pseudoinverse adversarial masks from the top positive eigenvector frame
- evaluate mask attacks against the paper's random-permutation baseline
- generate the Figure 7-style panel and attack curves

## Current Implementation

Files:

- `src/adversarial.py`
- `scripts/run_adversarial.py`
- `configs/baselines/mnist_noise015.yaml`
- `README.md` Task F section

The implementation follows the official repo's image-model decomposition
contract. For each target class, it takes the top `basis_size` positive
input-space eigenvectors, stacks them into `U in R^{D x K}`, and uses rows of
`pinv(U)` as selective "key" masks. Defaults match the paper: `basis_size=10`
and `attack_ranks=3`.

## Important Difference From The Old Attempt

The previous implementation built one matrix from each class's top-1
eigenvector. That is not the Figure 7 setup. Figure 7 uses the top positive
eigenvectors for a target digit and averages attack curves over the top three
masks.

The baseline is also no longer Gaussian noise. The paper baseline is a random
permutation of the adversarial mask, preserving the mask values and norm while
destroying the spatial/eigenvector alignment.

## Commands

```bash
# Evaluate all target digits; the visual panel defaults to target digit 3.
python scripts/run_adversarial.py \
  --checkpoint checkpoints/mnist_baseline_20260324-025128.pt

# Paper-text-focused version: only target digit 3.
python scripts/run_adversarial.py \
  --checkpoint checkpoints/mnist_baseline_20260324-025128.pt \
  --target-classes 3

# Figure 7B-style edge-only mask for unregularized models.
python scripts/run_adversarial.py \
  --checkpoint checkpoints/mnist_baseline_20260324-025128.pt \
  --target-classes 3 \
  --low-activity-mask

# Train the Figure 7A-style Gaussian-noise model first, then run the attack.
python scripts/train_baseline.py --config configs/baselines/mnist_noise015.yaml
python scripts/run_adversarial.py \
  --checkpoint checkpoints/<mnist_noise015_checkpoint>.pt \
  --target-classes 3
```

## Outputs

Outputs are saved under `results/adversarial/<checkpoint_stem>/`:

- `fig7_panel.png` - eigenvector / misclassified example / adversarial mask / permuted mask
- `success_curves.png` - accuracy and target-misclassification curves
- `masks.png` - compact mask-only panel
- `attack_metrics.json` - full per-mask and mean metrics
- `masks.pt` - masks and decomposition metadata

## Validation

Smoke-tested with:

```bash
/Users/longtenghai/opt/anaconda3/envs/web-env/bin/python scripts/run_adversarial.py \
  --checkpoint checkpoints/mnist_baseline_20260324-025128.pt \
  --target-classes 3 \
  --basis-size 10 \
  --attack-ranks 2 \
  --figure-ranks 1,2 \
  --figure-max-rows 2 \
  --magnitudes 0,1 \
  --max-batches 1 \
  --num-workers 0 \
  --device cpu \
  --output-dir /private/tmp/taskf_probe
```

The raw pseudoinverse check on the smoke output gives `pinv(U) @ U approx I`
for the selected target frame, with off-diagonal terms around `1e-7`.
