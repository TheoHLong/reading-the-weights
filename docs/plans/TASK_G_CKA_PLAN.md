# Task G CKA Transfer Plan (v2)

**Goal:** Adapt the CKA-based representational alignment method of Subramaniam et al. (2024) — "Training the Untrainable: Introducing Inductive Bias via Representational Alignment" — to a 1-layer bilinear MLP student. The paper's main results target deep FCNs/CNNs with many alignment points; we explicitly do not claim a faithful reproduction.

**Reference paper:** https://arxiv.org/abs/2410.20035
**Reference code:** https://github.com/vsubramaniam851/untrainable-networks (`rep_sim/`, `imagenet-comparison/image_class.py`)

**Dependency:** Phases 0–2 of `TASK_G_PLAN.md` (data.py CIFAR-10 support, baseline checkpoint, ResNet-18 teacher checkpoint) are complete.
**Downstream:** Task H consumes the post-CKA student checkpoint and compares its eigendecomposition against the baseline + KD checkpoints.
**Scope:** Linear CKA only, single transfer method. RSA, untrained-guide ablation, and `--early_stop` schedule are listed as optional extensions.

---

## Background

### Why CKA, not KD (intuition)

KD constrains the student's **output distribution** — it tells the student "your logits should match teacher's softmax." When the student's function class doesn't include teacher-like outputs (which is exactly our raw-pixel bilinear MLP situation: 45% ceiling vs. 96% teacher), the gradient direction is largely uninformative.

CKA constrains the student's **intermediate representations' relational structure**. For a batch of n samples, you compute the n×n similarity matrix among samples in each layer, and ask: does the student's per-layer similarity matrix look like the teacher's? This is a much weaker, more reachable constraint:

- It says nothing about absolute activation values.
- It only asks "if teacher thinks samples i and j are similar in layer L, you should think so too."
- It can be satisfied even when the student's function class is far from teacher's.

The paper's strongest result is that this works **even with a randomly initialized guide** — suggesting it's the architecture's relational structure (not the trained weights) that gets transferred.

### Limitations relative to the paper

This is important to state up front, because it constrains how we should interpret results.

The paper's main image experiments use a 48-layer narrow MLP (`ImageNetNarrowMLP`) and an 8192-wide shallow MLP, both with many natural hookable points. Their layer mapping spreads ResNet-50's ~17 alignment layers across many student layers, so each student layer is supervised roughly once.

Our student is a 1-layer bilinear MLP with only **two** non-trivial intermediate representations:

- `embed` output: post first linear projection from 3072 → 512
- `blocks.0` output: post bilinear multiplicative interaction (still 512-dim)

We therefore **cannot** apply the paper's recipe of "many-to-many uniform mapping with sum-reduction over a long chain of supervision points." Instead, we adapt the method:

- **explicit 1-to-1 mapping** between selected teacher and student layers (default below)
- **mean reduction** over layer pairs, so the loss scale is independent of mapping size
- α recalibrated to compensate the sum→mean change

These adaptations are noted because they affect what the result means. A weak gain here would not invalidate the paper — it would tell us "a 1-layer bilinear student lacks enough supervision surface area to absorb the conv inductive bias via this mechanism."

---

## Phase A — CKA primitives

### A.1 Add `src/cka.py`

Linear CKA via Hilbert-Schmidt Independence Criterion. Implementation follows `rep_sim/rep_sims.py` from the paper repo, but uses the Gram-matrix form throughout (numerically stable for our `n=256`, `d` ranging from 512 to 65536).

```python
from __future__ import annotations

import torch
from torch import Tensor


def _gram_centered(x: Tensor) -> Tensor:
    """Returns the centered linear Gram matrix K = X̃ X̃^T, where X̃ = X − col_mean(X)."""
    x = x - x.mean(dim=0, keepdim=True)
    return x @ x.T


def linear_cka(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Linear CKA between two activation matrices, both shaped (n_samples, n_features).

    Returns a scalar in [0, 1]: 1 = identical relational structure, 0 = unrelated.
    Differentiable in both arguments.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f'CKA requires matching batch sizes, got {x.shape[0]} vs {y.shape[0]}.')
    K = _gram_centered(x.float())
    L = _gram_centered(y.float())
    hsic_xy = (K * L).sum()
    hsic_xx = (K * K).sum()
    hsic_yy = (L * L).sum()
    return hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt() + eps)


def cka_distance(x: Tensor, y: Tensor) -> Tensor:
    """1 - CKA(x, y). This is what gets minimized."""
    return 1.0 - linear_cka(x, y)
```

**Why Gram form**: `K = X X^T` is `(n, n)`. With `n = 256`, every CKA call materializes two `256×256` matrices regardless of feature dim. For ResNet-18 layer1 output flattened to `(256, 65536)`, the alternative covariance form `X^T X` would be `65536×65536` ≈ 16 GB. Gram form sidesteps this entirely.

**Numerical caveat**: when activations have huge magnitude (e.g., post-ReLU conv features), `(K * K).sum()` can overflow in fp16. Cast to fp32 inside `linear_cka` (already done above). The paper's implementation also runs in fp32 for the same reason.

### A.2 Smoke test

Add `scripts/check_cka.py`:

```python
# Sanity checks
x = torch.randn(64, 128)
assert torch.isclose(linear_cka(x, x), torch.tensor(1.0), atol=1e-5)

y = torch.randn(64, 128)
score = linear_cka(x, y)
assert 0.0 <= score.item() <= 1.0

# Permutation invariance: shuffling features must not change CKA
perm = torch.randperm(128)
assert torch.isclose(linear_cka(x, x[:, perm]), torch.tensor(1.0), atol=1e-5)

# Scale invariance: scaling features must not change CKA
assert torch.isclose(linear_cka(x, 5.0 * x), torch.tensor(1.0), atol=1e-5)
```

---

## Phase B — Single activation-capture API

### B.1 `src/hooks.py`

`ActivationCapture` is the **single** API for getting intermediate activations from any model in this codebase. We deliberately do not maintain a parallel `get_guide_features()` function — two ways to do the same thing is a guaranteed source of drift bugs.

```python
from contextlib import contextmanager
from torch import nn

class ActivationCapture:
    """Forward-hook based activation capture. Single canonical API for both
    teacher and student in CKA training.

    Usage:
        with ActivationCapture(model, ['layer2', 'layer4']) as cap:
            _ = model(x)
            acts = cap.activations  # dict[str, Tensor]
    """
    def __init__(self, model: nn.Module, layer_names: list[str]) -> None:
        self.model = model
        self.layer_names = layer_names
        self.activations: dict[str, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self):
        named_modules = dict(self.model.named_modules())
        for name in self.layer_names:
            if name not in named_modules:
                raise KeyError(
                    f'Layer {name!r} not found in model. '
                    f'Available: {sorted(named_modules.keys())}'
                )
            handle = named_modules[name].register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self.activations[name] = out
        return hook
```

### B.2 Update `src/guide.py`

**Remove** the `get_guide_features()` stub. It was a placeholder anticipating a separate feature-extraction API, but `ActivationCapture` covers this use case cleanly. Any caller that wants teacher features just uses `ActivationCapture(teacher, [...])` directly.

### B.3 Layer name resolution

PyTorch's `named_modules()` returns names like `embed`, `blocks`, `blocks.0`, `blocks.0.gate`, `head`. The hookable names for our two architectures:

**Student (`BilinearImageClassifier`):**
- `embed` — post first linear projection, shape `(B, 512)`
- `blocks.0` — post bilinear multiplicative interaction, shape `(B, 512)`

**Teacher (CIFAR ResNet-18):**
- `layer1` — `(B, 64, 32, 32)`, low-level edges/textures
- `layer2` — `(B, 128, 16, 16)`, mid-level parts
- `layer3` — `(B, 256, 8, 8)`, object-level features
- `layer4` — `(B, 512, 4, 4)`, high-level semantic features

Verify available names with: `print(sorted(n for n, _ in model.named_modules() if n))`.

---

## Phase C — CKA training loop

### C.1 Layer mapping: explicit-first, uniform fallback

```python
# in transfer.py

def resolve_layer_map(
    transfer_cfg: dict,
    teacher_layers: list[str],
    student_layers: list[str],
) -> dict[str, str]:
    """Resolve the teacher→student layer mapping.

    Priority:
      1. transfer_cfg['layer_map'] — explicit dict, used as-is. Recommended.
      2. fallback: uniform spacing across teacher_layers and student_layers.
    """
    explicit = transfer_cfg.get('layer_map')
    if explicit:
        for t, s in explicit.items():
            if t not in teacher_layers:
                raise ValueError(f'layer_map references unknown teacher layer: {t}')
            if s not in student_layers:
                raise ValueError(f'layer_map references unknown student layer: {s}')
        return dict(explicit)

    # Uniform fallback (paper's recipe). Note this can map several teacher
    # layers to the same student layer when student is much shallower.
    n_t, n_s = len(teacher_layers), len(student_layers)
    if n_s == 0:
        return {}
    step = (n_s - 1) / max(n_t - 1, 1)
    return {teacher_layers[i]: student_layers[min(round(i * step), n_s - 1)]
            for i in range(n_t)}
```

### C.2 CKA loss with mean reduction (default)

```python
def cka_loss(
    student_acts: dict[str, Tensor],
    teacher_acts: dict[str, Tensor],
    layer_map: dict[str, str],
    reduction: str = 'mean',
) -> Tensor:
    """Sum or mean of (1 - linear_CKA) over all (teacher, student) layer pairs.

    Default 'mean' makes the loss scale invariant to the number of mapped pairs,
    so alpha keeps a stable meaning across different mappings. Use 'sum' to
    match the paper's exact reduction (only sensible when each student layer
    is supervised by ~1 teacher layer).
    """
    if reduction not in ('mean', 'sum'):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")

    if not layer_map:
        # No supervision: return zero on the student's autograd graph
        any_s = next(iter(student_acts.values()))
        return any_s.new_zeros(())

    pair_losses = []
    for t_name, s_name in layer_map.items():
        t = teacher_acts[t_name].reshape(teacher_acts[t_name].size(0), -1)
        s = student_acts[s_name].reshape(student_acts[s_name].size(0), -1)
        pair_losses.append(cka_distance(s, t))

    stacked = torch.stack(pair_losses)
    return stacked.mean() if reduction == 'mean' else stacked.sum()
```

### C.3 `train_cka_experiment()` in `transfer.py`

```python
def train_cka_experiment(config: dict) -> dict[str, Path]:
    set_seed(int(config['seed']))
    config.setdefault('train', {})
    config['train'].setdefault('split_seed', int(config['seed']))

    transfer_cfg = config.get('transfer', {})
    if transfer_cfg.get('method') != 'cka':
        raise ValueError("transfer.method must be 'cka' for train_cka_experiment().")

    dataset_bundle = build_image_dataloaders(config['dataset'], config['train'])
    device = resolve_device(config['train'].get('device', 'auto'))

    alpha = float(transfer_cfg.get('alpha', 3.0))   # see Hyperparameter notes
    reduction = transfer_cfg.get('reduction', 'mean')
    teacher_layers = list(transfer_cfg.get('teacher_layers', ['layer2', 'layer4']))
    student_layers = list(transfer_cfg.get('student_layers', ['embed', 'blocks.0']))
    early_stop_steps = transfer_cfg.get('early_stop_steps')  # None = always-on

    teacher = load_frozen_teacher(transfer_cfg['teacher_checkpoint'], device)
    student = build_image_classifier(config['model'], seed=int(config['seed'])).to(device)

    layer_map = resolve_layer_map(transfer_cfg, teacher_layers, student_layers)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(student.parameters(),
                      lr=float(config['train']['lr']),
                      weight_decay=float(config['train']['wd']))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config['train']['epochs']))

    # ... checkpoint setup identical to KD ...

    global_step = 0
    for epoch in tqdm(range(1, int(config['train']['epochs']) + 1), desc='training'):
        student.train()
        running_total_loss = 0.0
        running_ce_loss = 0.0
        running_cka_loss = 0.0
        running_correct = 0
        running_examples = 0

        for x, y in dataset_bundle.train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Teacher activations (detached, no_grad)
            with torch.no_grad():
                with ActivationCapture(teacher, teacher_layers) as t_cap:
                    _ = teacher(x)
                teacher_acts = {n: a.detach() for n, a in t_cap.activations.items()}

            # Student forward with autograd-live activation capture
            with ActivationCapture(student, student_layers) as s_cap:
                student_logits = student(x)
                student_acts = dict(s_cap.activations)

            ce = criterion(student_logits, y)

            use_cka = (early_stop_steps is None) or (global_step < early_stop_steps)
            if use_cka:
                rep = cka_loss(student_acts, teacher_acts, layer_map, reduction=reduction)
                loss = ce + alpha * rep
            else:
                rep = ce.new_zeros(())
                loss = ce

            loss.backward()
            optimizer.step()
            global_step += 1

            running_total_loss += loss.item() * y.size(0)
            running_ce_loss += ce.item() * y.size(0)
            running_cka_loss += rep.item() * y.size(0)
            running_correct += (student_logits.argmax(dim=-1) == y).sum().item()
            running_examples += y.size(0)

        scheduler.step()

        train_total_loss = running_total_loss / running_examples
        train_ce_loss = running_ce_loss / running_examples
        train_cka_loss = running_cka_loss / running_examples
        train_acc = running_correct / running_examples
        val_loss, val_acc = evaluate(student, dataset_bundle.val_loader, criterion, device)

        row = {
            'epoch': epoch,
            'train_total_loss': train_total_loss,
            'train_ce_loss': train_ce_loss,
            'train_cka_loss': train_cka_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0],
        }
        # ... save checkpoint with same schema as KD: model_state_dict, config, epoch, metrics=row ...
```

### C.4 Critical implementation details

**Hook lifetimes**: `ActivationCapture` is a context manager so hooks are removed every iteration. Leaking hooks across iterations causes silent memory growth and stale activations.

**Teacher in eval mode**: `load_frozen_teacher()` already calls `.eval()`. This means BN uses running statistics, **not** batch statistics. Crucial: if teacher BN ran in train mode, its activations would shift batch-to-batch and CKA gradients would be noisy.

**Don't `.detach()` student activations**: gradient must flow back through them. `cka_loss` keeps autograd live on the student side; teacher side is detached at capture time.

**val metrics stay pure CE**: model selection uses `val_acc` and `val_loss = CE(logits, labels)` only. No CKA term in validation. This keeps val_loss directly comparable to baseline and KD runs.

---

## Phase D — Configs

### `configs/transfer/cifar10_cka.yaml`

```yaml
experiment_name: cifar10_cka
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
  n_layer: 1
  d_output: 10
  bias: false
  residual: false

train:
  epochs: 50
  batch_size: 256
  lr: 0.001
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
  teacher_layers: [layer2, layer4]
  student_layers: [embed, blocks.0]
  layer_map:
    layer2: embed
    layer4: blocks.0
  # early_stop_steps: 300   # optional; uncomment to mimic paper's early-stop schedule
```

**Why this default mapping (1-to-1, layer2→embed, layer4→blocks.0)**: avoids double-counting (which the uniform 4→2 mapping would do), keeps loss scale stable across mapping changes, and is interpretable for Task H — "embed should encode mid-level features the way layer2 does; bilinear interaction should encode high-level features the way layer4 does."

**Why student/train hyperparameters are identical to baseline and KD configs**: the only intended variable across the three runs is the transfer term. Same lr, same wd, same epochs, same seed, same data split.

### `configs/transfer/cifar10_cka_debug.yaml`

Mirror of `cifar10_kd_debug.yaml`: 2 epochs, `max_train_batches: 2`, smaller batch. For smoke-testing the loop.

### `scripts/train_transfer.py` — single dispatch entry point

Extend the existing transfer entry point to dispatch on `transfer.method`:

```python
method = config.get('transfer', {}).get('method')
if method == 'kd':
    artifacts = train_kd_experiment(config)
elif method == 'cka':
    artifacts = train_cka_experiment(config)
else:
    raise ValueError(f'Unknown transfer.method: {method!r}')
```

This keeps `train_transfer.py` as the single transfer entry point for both KD and CKA.

---

## Phase E — Verification

### E.1 Smoke test the CKA primitive

```bash
python scripts/check_cka.py
```

Should print all sanity checks pass.

### E.2 Smoke test the training loop

```bash
python scripts/train_transfer.py --config configs/transfer/cifar10_cka_debug.yaml
```

Two epochs, 2 batches each. Confirm:
- Loop completes without hook errors.
- `metrics.csv` has all four columns: `train_total_loss`, `train_ce_loss`, `train_cka_loss`, `train_acc`.
- `train_cka_loss` is finite and in `[0, 1]` (with mean reduction).
- A checkpoint is saved with the KD-equivalent schema.

### E.3 Run the full experiment

```bash
python scripts/train_transfer.py --config configs/transfer/cifar10_cka.yaml
```

50 epochs. On a single CUDA GPU plan ~2× slower per step than KD due to extra hooks + teacher forward.

### E.4 Decompose and evaluate

```bash
python scripts/analyze_checkpoint.py --checkpoint checkpoints/cifar10_cka_<ts>.pt
python scripts/eval_checkpoint.py --checkpoint checkpoints/cifar10_cka_<ts>.pt
```

### E.5 Comparison table for Task H

| Model | val_acc | test_acc | top-k eigenvalue mass | Notes |
|-------|---------|----------|----------------------|-------|
| ResNet-18 teacher | 95.98% | (eval) | n/a | not bilinear |
| Bilinear baseline | 44.88% | (eval) | (decompose) | pre-transfer reference |
| Bilinear + KD | 45.10% | (eval) | (decompose) | logit-level transfer; marginal gain in this single configuration |
| Bilinear + CKA | ? | ? | ? | rep-level transfer (this phase) |

The headline question for Task H: **does the eigenvalue spectrum of the CKA-trained student concentrate on fewer top eigenvectors than baseline?** That would be direct evidence that representational alignment shifted bilinear weight structure toward conv-like inductive bias, even if accuracy itself doesn't move much.

---

## Files summary

| Action | Path | Phase |
|--------|------|-------|
| Create | `src/cka.py` | A |
| Create | `src/hooks.py` | B |
| Edit | `src/guide.py` (remove `get_guide_features` stub) | B |
| Edit | `src/transfer.py` (add `cka_loss`, `train_cka_experiment`, `resolve_layer_map`) | C |
| Edit | `scripts/train_transfer.py` (dispatch on `transfer.method`) | C |
| Create | `configs/transfer/cifar10_cka.yaml` | D |
| Create | `configs/transfer/cifar10_cka_debug.yaml` | D |
| Create | `scripts/check_cka.py` | E |

**Existing files NOT modified**: `model.py`, `decomposition.py`, `train.py`, `data.py`, `utils.py`, `analyze_checkpoint.py`, `eval_checkpoint.py`. The student checkpoint schema is unchanged, so all downstream consumers continue to work unmodified.

---

## Hyperparameter notes

- **`alpha = 3.0`** is the v2 default and is **not** the paper's `alpha = 1.0`. Reason: the paper sums `(1 - CKA)` over many layer pairs (rep_loss in `[0, ~4-17]`), while we use mean reduction over 1–2 pairs (rep_loss in `[0, 1]`). To keep the rep term roughly comparable to CE in early training, we need a larger α. If you switch `reduction: sum` to literally match the paper, drop α back to 1.0.

  Diagnostic: at the end of epoch 1, `alpha * train_cka_loss` should be within a factor of 2-3 of `train_ce_loss`. If it's much smaller, raise α; if it dominates from the start, lower α.

- **`layer_map`**: explicit `layer2 → embed`, `layer4 → blocks.0` is the v2 default. Alternatives worth trying:
  - `layer4 → blocks.0` only (single high-level supervision point — most direct test of "does deepest teacher feature transfer to bilinear interaction")
  - `[layer1, layer2, layer3, layer4] → [embed, embed, blocks.0, blocks.0]` (uniform fallback, more constraint but with double-counting on each student layer)

- **`reduction`**: `mean` is the v2 default. Use `sum` only if you switch to a 1-to-1 mapping with as many pairs as the paper has, or you want to deliberately let mapping size affect loss strength.

- **`early_stop_steps`**: paper drops the rep_sim term after 300 steps, arguing it just biases the optimization trajectory. For our 50-epoch CIFAR-10 run with batch=256, total steps ≈ 8800. 300 steps ≈ 1.7 epochs. Worth trying as an ablation: full-time CKA vs. CKA-then-CE-only.

---

## Open design choices

These are decisions worth confirming before implementation, not blockers:

1. **Linear CKA only, or also kernel CKA / RSA?** Paper supports `RSA` too (`rep_sim/rep_sims.py: DifferentiableRSA`). Linear CKA is faster and the paper's main results use it. Default to linear CKA only for v1; add RSA later if there's time.
2. **Untrained-guide ablation?** The paper's headline result is that random-init guide also helps. `load_frozen_teacher` could be extended with a `random_init: true` flag in the transfer config that builds the ResNet but skips the `load_state_dict`. Cheap to add; reveals whether the gain is from teacher's *knowledge* or from architecture's *prior structure*. Recommend including this as a stretch ablation if time permits.
3. **Should the head get any CKA pressure?** Default: no. Logit-level transfer was already tested via KD with marginal gain.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Hooks break under DataParallel/DDP | Low (we use single GPU) | Keep training on one device for v1 |
| Memory blow-up from holding teacher activations | Low | Teacher activations are detached and small (max 64 MB per layer at B=256); reused only within one step |
| CKA gradient is too weak to move bilinear weights meaningfully | Medium | Track `train_cka_loss` per epoch — if it doesn't fall, raise `alpha` or check hook wiring |
| Result still shows no accuracy improvement | Plausible | Spectrum-level changes (the actual Task H signal) can occur even when accuracy is flat. A null-but-clean result is also publishable: "rep-level transfer with 2 supervision points is insufficient for raw-pixel bilinear MLPs to absorb conv inductive bias under this configuration." Combined with KD's marginal gain, it strengthens the case for Ali's pooled architecture being a necessary precondition |
| Conclusions get over-interpreted | Medium | Use cautious language in plan and eventual writeup. Avoid "KD failed" / "CKA failed" — say "showed marginal gain in single tested configuration" or "did not produce a measurable accuracy gain" |

---

## Changelog

**v2 (current)** — incorporates 6 review-driven changes:

1. **Loss reduction**: default `cka_loss` reduction changed from `sum` to `mean`. Sum is still available via config. Reason: with our shallow student, naive sum makes loss scale depend on mapping size, breaking the meaning of α across configurations.
2. **Layer mapping**: default switched from uniform 4→2 to explicit 1-to-1 (`layer2 → embed`, `layer4 → blocks.0`). Avoids double-counting; more interpretable for Task H.
3. **α default**: raised from `1.0` to `3.0` to compensate the sum→mean scale change. Documented the conversion rule.
4. **Metrics decomposition**: training row now records `train_total_loss`, `train_ce_loss`, `train_cka_loss` separately. Same fix as the KD review (KD's mixed train_loss was misleading). val_loss stays pure CE.
5. **Single capture API**: `get_guide_features()` removed from `guide.py`. `ActivationCapture` is now the only path for intermediate-feature capture, used by both teacher and student in the training loop.
6. **Language tightening**:
   - "Faithfully following Subramaniam et al." → "Adapting Subramaniam et al.'s CKA guidance to a 1-layer bilinear student"
   - "KD failed" → "KD showed marginal gain in the single tested configuration"
   - Added "Limitations relative to the paper" section explicitly stating that our 2-layer supervision surface is much smaller than the paper's deep MLPs, so a weak result here would not invalidate the paper.

**v1** — initial CKA plan. Used uniform layer mapping with sum reduction, α=1.0, single train_loss column, parallel `get_guide_features` API. Superseded by v2.
