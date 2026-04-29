# Task G Deep Bilinear Plan (Path C)

**Goal:** Stack multiple bilinear blocks in the student so that CKA guidance has more supervision surface area, mimicking the paper's deep-MLP setup. Test whether deeper bilinear students absorb teacher inductive bias more effectively than the 1-layer student (which gave +1.74% CKA, +0.22% KD over baseline).

**Reference:** Subramaniam et al. (2024), "Training the Untrainable" — their narrow MLP has 48 layers and benefits strongly from CKA guidance with many alignment points.

**Dependency:** All earlier phases of `TASK_G_PLAN.md` and `TASK_G_CKA_PLAN.md` are complete. The teacher checkpoint, baseline (n_layer=1) checkpoint, KD checkpoint, and CKA (n_layer=1) checkpoint exist.

**Out of scope (explicitly deferred per user decision):**

- Multi-layer eigendecomposition machinery. `decomposition.py` line 32 currently raises on `n_layer > 1`. Task H consumers will not work on these checkpoints until that's extended. This plan **does not block on it** — we accept that deep-bilinear checkpoints are temporarily decomposition-incompatible. Extending decomposition is a separate work item.

---

## Background: why deeper might help (and where it might break)

### What changes when we stack bilinear layers

A single bilinear layer computes `(W_l x) ⊙ (W_r x)`, which is degree-2 in `x`. Stacking `n` layers naïvely gives a degree-`2^n` polynomial — for n=4 that's degree 16, for n=8 degree 256. **Pure stacked bilinears are pathologically hard to optimize.**

Two stabilizers that the paper's MLP gets for free (BN+ReLU+Dropout) but we currently don't:

1. **Residual connections** — `x = x + bilinear(x)` keeps each layer's contribution additive on top of an identity skip. The gradient highway is essential past depth ~2.
2. **A gating non-linearity** — `silu(W_l x) ⊙ (W_r x)` is the SwiGLU-style construction used in modern Transformers. It tames the multiplicative blowup and gives smooth gradients.

Our `Bilinear` class already supports both: `residual=True` is a flag on `BilinearImageClassifier`, and `gate='silu'` is an existing option on `Bilinear` (currently unused — `build_image_classifier` doesn't pass it).

### What more depth buys us for CKA

With `n_layer = N`, the student exposes `N + 1` natural CKA hookpoints: `embed`, `blocks.0`, `blocks.1`, …, `blocks.N-1`. At `N=4` we approach the paper's "many supervision layers" regime and can do near-1-to-1 mapping with ResNet-18's 4 BasicBlock stages. At `N=1` we had only 2 hookpoints and had to either double-count or pick 2 of teacher's 4 layers.

### What we explicitly accept losing

- **Single-layer decomposition story** — for n_layer > 1, `decomposition.py` raises. Task H downstream consumption is broken on these checkpoints until decomposition is extended. **User decision: defer extension; let deep checkpoints sit unused by Task H for now.**
- **Cleanest "interpretability" framing** — multi-layer bilinear is harder to reason about analytically. The "reading the weights" story per layer becomes per-block instead of model-wide.

---

## Architectural decisions for deep bilinear

### Required for stability

| Setting | n_layer=1 (current) | n_layer ≥ 2 (this plan) | Why |
|---------|---------------------|-------------------------|-----|
| `residual` | `false` | **`true`** | Gradient highway. Without this, deep bilinear stacks vanish/explode within 1-2 epochs |
| `gate` (in Bilinear) | unused (None) | **`'silu'`** (recommended) | SwiGLU-style gating tames multiplicative blowup. Optional if residual alone is enough but recommended for n_layer ≥ 4 |
| `bias` | `false` | `false` (keep) | No reason to change; keeps weight surgery clean |

### Tradeoff: pure bilinear (gate=None) vs. gated (gate='silu')

**Pure bilinear (gate=None):**
- Pros: cleaner mathematical structure, closer to "read the weights" thesis, decomposition machinery (when extended) won't need to model the gate
- Cons: Multiplicative depth blows up fast. Even with residual, n_layer ≥ 4 is iffy

**Gated bilinear (gate='silu'):**
- Pros: Trainable to arbitrary depth, matches how modern architectures actually use bilinear (SwiGLU, GLU)
- Cons: Decomposition story gets more complicated when revisited later (gate is a separate non-linearity, eigenvectors of raw bilinear weights no longer fully describe the model's function)

**Plan recommendation:** `residual=True, gate=None` for n_layer=2 (cleanest to interpret). If n_layer=2 fails to train cleanly, switch to `gate='silu'`. Promote `gate='silu'` to default for n_layer ≥ 4 regardless.

### Required code change in `model.py`

Currently `build_image_classifier` doesn't pass `gate` through:

```python
# model.py line 65 (current):
Bilinear(d_hidden, d_hidden, bias=bias) for _ in range(n_layer)
```

Need to plumb `gate` through `build_image_classifier`:

```python
def build_image_classifier(model_cfg: dict, seed: int) -> BilinearImageClassifier:
    return BilinearImageClassifier(
        d_input=int(model_cfg['d_input']),
        d_hidden=int(model_cfg['d_hidden']),
        d_output=int(model_cfg['d_output']),
        n_layer=int(model_cfg['n_layer']),
        bias=bool(model_cfg['bias']),
        residual=bool(model_cfg['residual']),
        gate=model_cfg.get('gate'),       # NEW
        seed=seed,
    )
```

And `BilinearImageClassifier.__init__` needs the same plumbing:

```python
self.blocks = nn.ModuleList([
    Bilinear(d_hidden, d_hidden, bias=bias, gate=gate) for _ in range(n_layer)
])
```

This is the **only** edit to `model.py`. Backward compatible: existing configs without `gate` keep the current behavior (gate=None).

---

## Phase A — n_layer=2 baseline (sanity)

Before turning on CKA, train a 2-layer bilinear baseline. We need this for two reasons: (1) confirm the architectural changes (residual, optional gate) train cleanly without transfer, (2) have a fair comparison point — CKA at depth 2 must beat baseline at depth 2, not baseline at depth 1.

### A.1 Edit `model.py`

Plumb `gate` through `build_image_classifier` and `BilinearImageClassifier` as shown above.

### A.2 Add `configs/baselines/cifar10_baseline_n2.yaml`

```yaml
experiment_name: cifar10_baseline_n2
seed: 42
output_dir: results/metrics
checkpoint_dir: checkpoints

dataset:
  name: cifar10
  root: data/raw
  image_size: 32
  channels: 3
  num_classes: 10

model:
  d_input: 3072
  d_hidden: 512
  n_layer: 2          # changed
  d_output: 10
  bias: false
  residual: true      # changed (essential for stability)
  gate: null          # explicit — no gating for first attempt

train:
  epochs: 50
  batch_size: 256
  lr: 0.0005          # halved from n_layer=1; deeper nets are touchier
  wd: 0.01
  num_workers: 2
  pin_memory: true
  device: auto
  val_fraction: 0.1
```

### A.3 Run

```bash
python scripts/train_baseline.py --config configs/baselines/cifar10_baseline_n2.yaml
```

### A.4 Acceptance criteria

- Loop completes without NaN or gradient explosion
- `train_loss` strictly decreases over the first 5 epochs (not flat, not exploding)
- `val_acc` reaches at least ~40% by epoch 50 (n_layer=1 baseline is 44.88%; deeper without transfer might lose a couple points before residual fully helps)

### A.5 Failure mode: training is unstable

If `train_loss` goes NaN, oscillates wildly, or `val_acc` stays at random (~10%):

1. First try `lr: 0.0001` (5x lower)
2. Then add `gate: silu` to the config
3. Then check that `residual: true` is actually being read (print model on init)

Don't proceed to CKA until baseline n_layer=2 trains cleanly. Otherwise CKA-vs-baseline comparison is contaminated by training instability.

---

## Phase B — n_layer=2 + CKA

### B.1 Add `configs/transfer/cifar10_cka_n2.yaml`

```yaml
experiment_name: cifar10_cka_n2
seed: 42
output_dir: results/metrics
checkpoint_dir: checkpoints

dataset:
  name: cifar10
  root: data/raw
  image_size: 32
  channels: 3
  num_classes: 10

model:
  d_input: 3072
  d_hidden: 512
  n_layer: 2
  d_output: 10
  bias: false
  residual: true
  gate: null         # match the baseline config exactly

train:
  epochs: 50
  batch_size: 256
  lr: 0.0005          # match baseline
  wd: 0.01
  num_workers: 2
  pin_memory: true
  device: auto
  val_fraction: 0.1

transfer:
  method: cka
  teacher_checkpoint: checkpoints/resnet18_cifar10_teacher.pt
  alpha: 3.0
  reduction: mean
  teacher_layers: [layer1, layer2, layer3, layer4]
  student_layers: [embed, blocks.0, blocks.1]
  layer_map:
    layer1: embed
    layer2: blocks.0
    layer3: blocks.0
    layer4: blocks.1
```

**Layer-mapping rationale at n_layer=2:**
- 4 teacher layers, 3 student hookpoints → can't 1-to-1
- Suggested: teacher's first stage → embed (raw-pixel level), teacher's middle stages → blocks.0 (first bilinear interaction), teacher's deepest stage → blocks.1 (second bilinear interaction)
- This puts the most semantic supervision (`layer4`) on the deepest student layer. Worth experimenting with `layer4 → blocks.0` instead if first attempt is weak.

### B.2 Run + verify

```bash
python scripts/train_transfer.py --config configs/transfer/cifar10_cka_n2.yaml
```

Acceptance: `train_cka_loss` decreases, and `val_acc` beats Phase A's baseline_n2 by **at least 2%** (i.e., depth gives CKA more room than n_layer=1's +1.74% gain).

If the gain at n_layer=2 is still ≤ +2% over baseline_n2, the bottleneck is likely capacity per layer, not number of supervision points → proceed to Phase C with more depth.

---

## Phase C — n_layer=4 (paper-shape)

This is where the paper's "many supervision layers" benefit should actually start showing up. With 4 student blocks plus embed = 5 hookpoints, we get near-1-to-1 mapping with teacher's 4 BasicBlock stages.

### C.1 Configs

`configs/baselines/cifar10_baseline_n4.yaml`:

```yaml
experiment_name: cifar10_baseline_n4
# ... same as baseline_n2 except: ...
model:
  n_layer: 4
  residual: true
  gate: silu        # promote to silu at n_layer ≥ 4
train:
  lr: 0.0003        # lower again for deeper net
  epochs: 75        # more depth → more epochs to converge
```

`configs/transfer/cifar10_cka_n4.yaml`:

```yaml
experiment_name: cifar10_cka_n4
# ... same model + train as baseline_n4 ...
transfer:
  method: cka
  teacher_checkpoint: checkpoints/resnet18_cifar10_teacher.pt
  alpha: 3.0
  reduction: mean
  teacher_layers: [layer1, layer2, layer3, layer4]
  student_layers: [embed, blocks.0, blocks.1, blocks.2, blocks.3]
  layer_map:
    layer1: embed
    layer2: blocks.0
    layer3: blocks.1
    layer4: blocks.3       # skip blocks.2 to get full coverage
```

**Mapping rationale at n_layer=4:** teacher's 4 BasicBlock stages map roughly 1-to-1 onto student's `embed/blocks.0/blocks.1/blocks.3`. `blocks.2` is left unsupervised — it acts as "free capacity" that the model can use for things teacher doesn't constrain. Worth A/B testing against a fully-mapped variant later (`layer4 → blocks.2; layer4 → blocks.3` would double-supervise the deep end).

### C.2 Acceptance

The headline expectation: **CKA n_layer=4 beats baseline n_layer=4 by more than CKA n_layer=2 beats baseline n_layer=2**. If this scaling shows up, the paper's "deeper student → bigger CKA gain" story reproduces in our setting.

If gain doesn't scale with depth, the limiting factor is something else (likely raw-pixel input lacking spatial structure regardless of capacity), and we should pivot to Path D (Ali's pooled architecture) rather than going deeper.

---

## Phase D — n_layer=8 (stretch, optional)

Only attempt if Phase C shows clear depth-scaling of CKA gain.

### D.1 Configs

`configs/baselines/cifar10_baseline_n8.yaml` and `configs/transfer/cifar10_cka_n8.yaml`:

- `n_layer: 8`
- `gate: silu` (mandatory at this depth)
- `residual: true` (mandatory)
- `lr: 0.0001`
- `epochs: 100`
- For CKA, layer map: teacher's 4 stages → student blocks `[0, 2, 4, 7]` (uniform spacing)

### D.2 Risk acceptance

n_layer=8 with multiplicative-degree dynamics is genuinely hard to train even with residual+gate. If training is unstable at this depth, that's an interesting finding in itself ("bilinear depth has practical limits") but not a blocker for the paper.

---

## Phase E — Verification and comparison

### E.1 Result table

| Model | val_acc | Δ over n=1 baseline | CKA Δ over same-depth baseline |
|-------|---------|---------------------|-------------------------------|
| Baseline n=1 | 44.88% (existing) | — | — |
| Baseline n=2 | (Phase A) | (compute) | — |
| CKA n=2 | (Phase B) | (compute) | (compute) |
| Baseline n=4 | (Phase C) | (compute) | — |
| CKA n=4 | (Phase C) | (compute) | (compute) |
| Baseline n=8 | (Phase D, optional) | — | — |
| CKA n=8 | (Phase D, optional) | — | (compute) |

### E.2 Test-set evaluation

For every checkpoint:

```bash
python scripts/eval_checkpoint.py --checkpoint checkpoints/<name>.pt
```

`eval_checkpoint.py` already handles the bilinear student via `build_image_classifier`. No changes needed — `n_layer` is read from the checkpoint's config.

### E.3 Diagnostics for CKA runs

Same diagnostics as the n_layer=1 CKA run:

- `train_cka_loss` finite + trending down
- `α · train_cka_loss` and `train_ce_loss` within ~3× of each other in early training
- val_loss / val_acc not collapsing

If these diagnostics fail at deeper n_layer, **stop and debug before scaling further**. A broken CKA run at n=4 doesn't help the comparison.

---

## Files summary

| Action | Path | Phase |
|--------|------|-------|
| Edit | `src/model.py` (plumb `gate` through builder) | A |
| Create | `configs/baselines/cifar10_baseline_n2.yaml` | A |
| Create | `configs/transfer/cifar10_cka_n2.yaml` | B |
| Create | `configs/baselines/cifar10_baseline_n4.yaml` | C |
| Create | `configs/transfer/cifar10_cka_n4.yaml` | C |
| Create | `configs/baselines/cifar10_baseline_n8.yaml` | D (optional) |
| Create | `configs/transfer/cifar10_cka_n8.yaml` | D (optional) |

**Files NOT modified:** `data.py`, `train.py`, `transfer.py`, `guide.py`, `cka.py`, `hooks.py`, `eval_checkpoint.py`, `analyze_checkpoint.py`. The whole training/transfer pipeline already supports arbitrary `n_layer` via the `model.n_layer` config field; only the gate plumbing in `model.py` is missing.

`decomposition.py` line 32 will continue to raise on these checkpoints. **Accepted; Task H consumers will not be able to use deep checkpoints until decomposition is extended in a separate work item.**

---

## Hyperparameter notes

- **`lr` schedule across depths**: empirical recipe — halve `lr` each time `n_layer` doubles. n=1 used `lr=1e-3`; n=2 → `5e-4`; n=4 → `~3e-4`; n=8 → `1e-4`. Adjust based on early epoch behavior.
- **`epochs` across depths**: deeper nets need more epochs to converge under the same lr schedule. n=2 stays at 50; n=4 bump to 75; n=8 bump to 100.
- **`alpha` for CKA**: keep at 3.0 across depths. The mean reduction in `cka_loss` keeps scale stable. If you change to sum, recalibrate.
- **`gate`**: null for n=1,2 (cleaner story); silu for n≥4 (training stability). Always use the same gate setting in matched baseline + CKA configs.
- **Initialization**: `BilinearImageClassifier.__init__` uses `torch.manual_seed(seed)` then default Linear init. This is generally OK for residual networks at moderate depth but may need scaled init at n_layer=8. Watch first-epoch train_loss — if it starts > log(num_classes) ≈ 2.3 by a lot, init is too large.

---

## Open design choices

1. **Skip baseline retraining at each depth?** No — we need baseline at each depth for fair comparison. Can't compare CKA n_layer=4 against baseline n_layer=1; the architectural change alone could be a confounder.

2. **Should `head` be a CKA target?** No, same reasoning as before — logit-level transfer was already tested via KD with marginal gain.

3. **Embedding alignment at very deep nets?** At n=8, mapping `layer1 → embed` may over-constrain the very first layer. Consider dropping embed from supervision when n ≥ 8 and let the model learn the input projection freely.

4. **Multi-seed?** Single seed is consistent with what we did for n_layer=1. Multi-seed would strengthen any observed depth-scaling effect statistically. Recommend: if depth scaling appears in single-seed runs, run a quick 3-seed confirmation at n_layer=2 and n_layer=4 before declaring victory.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Training unstable at n_layer ≥ 2 even with residual | Medium | Phase A is exactly the gate; don't proceed to CKA until baseline trains cleanly. Fallback: enable `gate='silu'` |
| CKA gain doesn't scale with depth | Medium | This is itself a finding ("bilinear depth alone doesn't unlock conv inductive bias"). Pivot to Path D (Ali's pooled architecture) |
| Decomposition can't be extended in time for Task H | High (per user, this is deferred) | Keep n_layer=1 CKA checkpoint as the "Task H deliverable"; deep checkpoints are an "extra experiment" for the report's discussion section, not the main eigendecomposition figure |
| Compute budget blows up | Medium at n_layer=8 | n_layer=8 is explicitly optional (Phase D). Stop at Phase C if time-constrained — the n=2 vs n=4 comparison alone tests the depth-scaling hypothesis |
| All deep CKA runs converge to the same plateau as n=1 | Plausible | Would be evidence that the architectural prior gap (no convolution / no spatial locality) cannot be closed by representational pressure alone, regardless of student depth or supervision density. Combined with the existing n=1 KD/CKA results, this is a publishable null result |

---

## Decision log (deferred items)

- **Multi-layer eigendecomposition** — deferred per user; will be addressed separately. Deep checkpoints are not Task H deliverables under this plan.
- **Untrained-guide ablation** (paper's headline) — could overlay onto any depth in this plan; deferred to keep this plan focused on depth-scaling alone.
- **RSA distance instead of CKA** — same comment.
- **Pooled architecture (Ali's path)** — explicitly an alternative to this plan, not a follow-up. If depth-scaling fails to materialize at n=4, pooled is the recommended pivot.

---

## Changelog

- **v1**: initial plan for Path C. Stays consistent with KD and 1-layer CKA infrastructure (no changes to `transfer.py`, `cka.py`, `hooks.py`). Only model.py needs a small plumbing edit. Deep checkpoints are accepted as decomposition-incompatible for now; user will extend decomposition in a separate work item.
