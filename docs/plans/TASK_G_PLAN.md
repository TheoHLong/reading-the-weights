# Task G Implementation Plan (v3)

**Goal:** Implement inductive bias transfer training logic (Knowledge Distillation) + set up guide network (ResNet-18 on CIFAR-10).

**Dependency:** Task A (complete)
**Downstream:** Task H consumes the checkpoints produced here to compare eigenvector spectra before vs. after transfer.

**Context from Task D (Ali):** Raw-pixel bilinear MLP on CIFAR-10 trains to ~44% test acc but stays broad-spectrum (needs rank 64–128+ to recover full performance). A fixed 4×4 average-pooling preprocessor yields much better spectral behavior. We don't have Ali's code, so we implement CIFAR-10 support ourselves.

**Scope:**
- KD only. CKA-guided training is deferred — KD already produces the pre/post-transfer checkpoint pair that Task H needs.
- **Pooled variant (4×4 avg-pool preprocessor) is explicitly OUT of scope for Task G.** Adding it would introduce a second intervention variable on top of KD, turning the comparison into a 2×2 design (raw/no-KD, raw/KD, pooled/no-KD, pooled/KD). Task H's clean signal requires holding everything constant except KD. Pooled variant is a follow-up extension, not part of this milestone.

---

## Phase 0 — 扩展 data.py 支持 CIFAR-10

### Problem with naive approach

Current `data.py` (line 38–52) creates one dataset instance, then `random_split` it into train/val subsets. Both subsets share the same transform. If we attach augmentation (RandomCrop, HorizontalFlip) to this single instance, **val data would also be augmented**, which corrupts validation metrics.

### Solution: train/eval transform separation with shared split indices

**Edit `src/data.py`:**

1. Add `'cifar10': datasets.CIFAR10` to `DATASETS`.
2. Define CIFAR-10 normalization constants at **module top level** (these are the single source of truth — teacher and student both `from src.data import CIFAR10_MEAN, CIFAR10_STD`; never duplicate these literals anywhere else in the codebase):
   ```python
   CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
   CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
   ```
3. For CIFAR-10, construct **two** dataset objects with different transforms:
   ```python
   train_transform = Compose([
       RandomCrop(32, padding=4),
       RandomHorizontalFlip(),
       ToTensor(),
       Normalize(CIFAR10_MEAN, CIFAR10_STD),
   ])
   eval_transform = Compose([
       ToTensor(),
       Normalize(CIFAR10_MEAN, CIFAR10_STD),
   ])

   train_dataset_aug  = CIFAR10(root, train=True, transform=train_transform)
   train_dataset_eval = CIFAR10(root, train=True, transform=eval_transform)
   ```
4. Generate split indices **once**, then apply to each dataset:
   ```python
   indices = torch.randperm(num_examples, generator=torch.Generator().manual_seed(split_seed))
   train_idx = indices[:num_train].tolist()
   val_idx   = indices[num_train:].tolist()

   train_subset = Subset(train_dataset_aug, train_idx)
   val_subset   = Subset(train_dataset_eval, val_idx)
   ```
5. Test dataset also uses `eval_transform`.
6. For MNIST/FMNIST, keep the current code path unchanged (ToTensor only, single dataset instance, `random_split`). This avoids touching any existing behavior.

**Create `configs/baselines/cifar10_baseline.yaml`:**
```yaml
experiment_name: cifar10_baseline
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
  d_input: 3072      # 32 * 32 * 3
  d_hidden: 512
  n_layer: 1
  d_output: 10
  bias: false
  residual: false

train:
  epochs: 50
  batch_size: 256
  lr: 0.001
  wd: 0.01          # bilinear MLP; lighter than MNIST's 0.5
  num_workers: 2
  pin_memory: true
  device: auto
```

### Why no model changes are needed

`BilinearImageClassifier.forward()` calls `x.flatten(start_dim=1)`, so 3×32×32 → 3072 automatically.

### Verification

- `python scripts/smoke_test.py --config configs/baselines/cifar10_baseline.yaml` completes without error.
- Confirm val_subset uses eval_transform (no random augmentation) by inspecting a sample.

---

## Phase 1 — 训练 CIFAR-10 baseline 学生

### What to do

1. Run `python scripts/train_baseline.py --config configs/baselines/cifar10_baseline.yaml`.
2. Record `best_val_acc` — expected ~40–45% based on Ali's results with raw-pixel input.
3. Run `python scripts/analyze_checkpoint.py --checkpoint checkpoints/<cifar10_best>.pt`.

### Artifacts produced

- `checkpoints/cifar10_baseline_<timestamp>.pt` — pre-transfer student checkpoint
- `results/analysis/cifar10_baseline_<timestamp>/decomposition.pt` — baseline eigenvectors

---

## Phase 2 — 准备 ResNet-18 guide network

### What to do

1. **Create `src/guide.py`** — only the parts KD needs:

   - `build_cifar_resnet18(num_classes=10) -> nn.Module`:
     - Start from `torchvision.models.resnet18(weights=None)`.
     - Replace `conv1` with `nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)`.
     - Replace `maxpool` with `nn.Identity()`.
     - Replace `fc` with `nn.Linear(512, num_classes)`.
   - `load_frozen_teacher(checkpoint_path, device) -> nn.Module`:
     - Load checkpoint, call `model.eval()` + `model.requires_grad_(False)`.
     - Return frozen model.
   - `get_guide_features()` — **stub only**, leave the signature for future CKA. Do not implement or block on it.

2. **Create `scripts/train_guide.py`** — standalone training script for ResNet-18:
   - **Does not reuse `train_image_experiment()`** — the teacher has a completely different optimizer recipe.
   - Uses SGD (not AdamW), lr=0.1, momentum=0.9, wd=5e-4, CosineAnnealingLR.
   - **Imports `CIFAR10_MEAN` and `CIFAR10_STD` from `src.data`** — never re-define these literals here. Student and teacher must see the same preprocessed pixels; duplicated constants are a silent-drift hazard.
   - After training completes, save **two artifacts**:
     - `checkpoints/resnet18_cifar10_<timestamp>.pt` — provenance copy (lets you compare multiple teacher runs later).
     - `checkpoints/resnet18_cifar10_teacher.pt` — stable alias (referenced by downstream configs; just `shutil.copy()` the best timestamped checkpoint).
   - Saves checkpoint in a format **that includes the full config**, so `eval_checkpoint.py` can rebuild dataloader/device uniformly for both student and teacher (no special-case branches):
     ```python
     {
         'model_state_dict': OrderedDict,
         'config': dict,        # at minimum: config.dataset and config.train
         'num_classes': 10,
         'best_val_acc': float,
         'epoch': int,
     }
     ```

3. **Create `configs/guides/resnet18_cifar10.yaml`:**
   ```yaml
   experiment_name: resnet18_cifar10
   seed: 42

   dataset:
     name: cifar10
     root: data/raw
     image_size: 32
     channels: 3
     num_classes: 10

   train:
     epochs: 200
     batch_size: 128
     lr: 0.1
     momentum: 0.9
     wd: 0.0005
     num_workers: 2
     pin_memory: true
     device: auto
   ```

### Design notes

- The guide is NOT a `BilinearImageClassifier`. It never goes through `decompose_bilinear_model()`. It only provides logits.
- ~93% val_acc is an empirical expectation, not an acceptance criterion. Anything above ~90% should provide a meaningful teaching signal.

### Artifacts produced

- `checkpoints/resnet18_cifar10_teacher.pt` — stable alias, referenced by KD config

---

## Phase 3 — 实现 Knowledge Distillation 训练

This is the core of Task G.

### What to do

1. **Create `src/transfer.py`:**

   - `kd_loss(student_logits, teacher_logits, labels, alpha, temperature) -> Tensor`:
     ```python
     ce = F.cross_entropy(student_logits, labels)
     kl = F.kl_div(
         F.log_softmax(student_logits / T, dim=-1),
         F.softmax(teacher_logits / T, dim=-1),
         reduction='batchmean',
     )
     return alpha * ce + (1 - alpha) * T * T * kl
     ```

   - `train_kd_experiment(config) -> dict[str, Path]`:
     - Mirrors the structure of `train_image_experiment()` in `train.py`.
     - Loads frozen teacher via `load_frozen_teacher()`.
     - Each training step: forward both teacher and student on the same batch, compute `kd_loss`.
     - Student uses AdamW (same as baseline) — we want the **only** difference to be the KD loss, so results are a clean comparison.
     - Saves checkpoints in the **exact same format** as `train.py`:
       ```python
       {
           'model_state_dict': OrderedDict,
           'config': dict,
           'epoch': int,
           'metrics': dict,   # {'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'}
       }
       ```
       This ensures `decompose_bilinear_model()` and `analyze_checkpoint.py` work unchanged. Task H depends on this.

2. **Create `scripts/train_transfer.py`** — entry point that calls `train_kd_experiment()`.

3. **Create `configs/transfer/cifar10_kd.yaml`:**
   ```yaml
   experiment_name: cifar10_kd
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

   transfer:
     method: kd
     teacher_checkpoint: checkpoints/resnet18_cifar10_teacher.pt   # stable alias
     alpha: 0.5
     temperature: 4.0
   ```

### Hyperparameter notes

- `alpha`: 0.5 is a safe starting point. If transfer effect is weak, try 0.3 (more weight on teacher).
- `temperature`: 4.0 is standard. Higher T → softer distributions → student learns more inter-class structure. Try 3–6.
- Student train config (lr, wd, epochs) should match Phase 1 baseline exactly, so the only variable is KD vs. no-KD.

---

## Phase 4 — 端到端验证

### 4a. Decomposition check

Run `python scripts/analyze_checkpoint.py --checkpoint checkpoints/<cifar10_kd_best>.pt` and confirm `decomposition.pt` generates without error.

### 4b. Held-out test evaluation

**Create `scripts/eval_checkpoint.py`** — a small script that:
1. Loads a student checkpoint (baseline or KD).
2. Builds test_loader via `build_image_dataloaders()`.
3. Runs `evaluate(model, test_loader, criterion, device)`.
4. Prints test_loss and test_acc.

Run on all three models to produce a clean results table:

| Model | test_acc |
|-------|----------|
| ResNet-18 teacher | ~93% |
| Bilinear baseline (no transfer) | ~44% |
| Bilinear + KD | ? |

### 4c. Eigenvalue sanity check

Compare eigenvalue distributions between baseline and KD checkpoints. The spectra should visibly differ — this is the signal Task H will analyze.

### Artifacts for Task H handoff

| Artifact | Purpose |
|----------|---------|
| `checkpoints/cifar10_baseline_<ts>.pt` | Pre-transfer student |
| `checkpoints/cifar10_kd_<ts>.pt` | Post-transfer student |
| `checkpoints/resnet18_cifar10_teacher.pt` | Frozen teacher |
| `results/analysis/cifar10_baseline_<ts>/decomposition.pt` | Pre-transfer eigenvectors |
| `results/analysis/cifar10_kd_<ts>/decomposition.pt` | Post-transfer eigenvectors |

---

## Files summary

| Action | Path | Phase |
|--------|------|-------|
| Edit | `src/data.py` | 0 |
| Create | `configs/baselines/cifar10_baseline.yaml` | 0 |
| Create | `src/guide.py` | 2 |
| Create | `configs/guides/resnet18_cifar10.yaml` | 2 |
| Create | `scripts/train_guide.py` | 2 |
| Create | `src/transfer.py` | 3 |
| Create | `scripts/train_transfer.py` | 3 |
| Create | `configs/transfer/cifar10_kd.yaml` | 3 |
| Create | `scripts/eval_checkpoint.py` | 4 |

**Existing code that should NOT be modified:** `model.py`, `decomposition.py`, `train.py`, `utils.py`.

---

## Dependency graph

```
Phase 0 (data.py + cifar10 config)     Phase 2 (guide.py + ResNet-18 training)
         |                                        |
         v                                        |
Phase 1 (baseline student training)               |
         |                                        |
         +--------------------+-------------------+
                              |
                              v
                   Phase 3 (KD training)
                              |
                              v
                   Phase 4 (verify + test eval)
```

Phase 0 and Phase 2 can run in parallel. Phase 3 needs both. Phase 4 needs Phase 3.

---

## Changelog from v1

1. **Phase 0:** Fixed val augmentation leak. Now uses two dataset instances (train_transform / eval_transform) with shared split indices, instead of random_split on a single augmented dataset.
2. **Phase 2:** Teacher uses its own optimizer recipe (SGD, lr=0.1, wd=5e-4) instead of reusing student config. ~93% is an empirical expectation, not a hard criterion. Teacher checkpoint saved with a stable alias to avoid timestamp coupling.
3. **Phase 3:** `teacher_checkpoint` path now points to stable alias `resnet18_cifar10_teacher.pt`. Student and teacher share the same CIFAR-10 normalization constants.
4. **Phase 4:** Added `scripts/eval_checkpoint.py` for held-out test evaluation on all models, producing a clean comparison table.
5. **Scope:** Removed CKA from scope. `get_guide_features()` is a stub only.
