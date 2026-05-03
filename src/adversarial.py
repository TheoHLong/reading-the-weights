"""Task F: Figure 7 adversarial masks from bilinear eigenvectors.

The official image model exposes the important primitive through
``Model.decompose()``: for a single-layer classifier it returns eigenvectors of
the symmetrised class interaction matrices, projected back to input space.  The
Figure 7 attack builds "keys" for these eigenvectors with a pseudoinverse.

For each target class, we stack its top positive input-space eigenvectors as
columns of ``U`` and take rows of ``pinv(U)``.  Row ``i`` is the mask that
selectively activates eigenvector ``i`` in that class-specific frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from src.decomposition import (
    build_bilinear_tensor,
    project_eigenvectors_to_input,
    symmetrize_bilinear_tensor,
)
from src.model import BilinearImageClassifier


@dataclass
class AdversarialMasks:
    """Masks and metadata for Figure 7-style attacks.

    Shapes:
        masks_pinv         [A, D]  unit-L2 pseudoinverse masks used for attack
        masks_pinv_raw     [A, D]  raw pseudoinverse rows before normalisation
        masks_eigvec       [A, D]  unit-L2 source eigenvectors for comparison
        target_classes     [A]     target class for each attack mask
        eigen_ranks        [A]     1-indexed positive-eigenvector rank
        eigvals_selected   [A]     eigenvalue for each source eigenvector
        basis_vectors      [C, K, D] unit-L2 vectors used to build each pinv
        basis_eigvals      [C, K]  eigenvalues paired with ``basis_vectors``
    """

    masks_pinv: Tensor
    masks_pinv_raw: Tensor
    masks_eigvec: Tensor
    target_classes: Tensor
    eigen_ranks: Tensor
    eigvals_selected: Tensor
    basis_vectors: Tensor
    basis_eigvals: Tensor


@dataclass
class AttackEvaluation:
    """Metrics for a set of targeted attack masks."""

    magnitudes: Tensor                  # [M]
    target_success_by_mask: Tensor      # [M, A]
    accuracy_by_mask: Tensor            # [M, A]
    target_success_mean: Tensor         # [M]
    accuracy_mean: Tensor               # [M]
    clean_acc: float
    num_examples: int


def _normalise_rows(x: Tensor, eps: float = 1e-12) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _as_class_tensor(
    target_classes: Sequence[int] | Tensor | None,
    num_classes: int,
    device: torch.device,
) -> Tensor:
    if target_classes is None:
        return torch.arange(num_classes, device=device, dtype=torch.long)
    if isinstance(target_classes, Tensor):
        classes = target_classes.to(device=device, dtype=torch.long)
    else:
        classes = torch.tensor(list(target_classes), device=device, dtype=torch.long)
    if classes.numel() == 0:
        raise ValueError('target_classes must contain at least one class.')
    if int(classes.min()) < 0 or int(classes.max()) >= num_classes:
        raise ValueError(f'target_classes must be in [0, {num_classes - 1}].')
    return classes


def _top_positive_indices(eigvals: Tensor, count: int) -> Tensor:
    """Return indices of the largest positive eigenvalues.

    If a class has fewer than ``count`` positive values, fall back to the
    largest remaining eigenvalues so the caller still gets a fixed-size basis.
    """
    order = torch.argsort(eigvals, descending=True)
    positive = order[eigvals[order] > 0]
    if positive.numel() >= count:
        return positive[:count]
    remaining = order[~torch.isin(order, positive)]
    return torch.cat([positive, remaining[: count - positive.numel()]])


@torch.no_grad()
def compute_adversarial_masks(
    model: BilinearImageClassifier,
    *,
    target_classes: Sequence[int] | Tensor | None = None,
    basis_size: int = 10,
    attack_ranks: int = 3,
    normalise_eigenvectors: bool = True,
    normalise_masks: bool = True,
) -> AdversarialMasks:
    """Build Figure 7 pseudoinverse masks for a single-layer bilinear model.

    ``basis_size`` corresponds to the paper's top-10 positive eigenvector frame.
    ``attack_ranks`` controls how many rows of that frame are evaluated; the
    paper reports curves averaged over the top three.
    """
    if len(model.blocks) != 1:
        raise ValueError(
            'Adversarial mask construction supports single-layer bilinear '
            'models only, matching the decomposition pipeline.'
        )
    if basis_size <= 0:
        raise ValueError('basis_size must be positive.')
    if attack_ranks <= 0:
        raise ValueError('attack_ranks must be positive.')
    if attack_ranks > basis_size:
        raise ValueError('attack_ranks cannot exceed basis_size.')

    bilinear_tensor = build_bilinear_tensor(model)
    symmetrized = symmetrize_bilinear_tensor(bilinear_tensor)
    eigvals, eigvecs_hidden = torch.linalg.eigh(symmetrized)
    num_classes = eigvals.shape[0]
    classes = _as_class_tensor(target_classes, num_classes, eigvals.device)

    selected_hidden: list[Tensor] = []
    selected_vals: list[Tensor] = []
    for c in range(num_classes):
        idx = _top_positive_indices(eigvals[c], basis_size)
        selected_hidden.append(eigvecs_hidden[c, :, idx].T)
        selected_vals.append(eigvals[c, idx])

    hidden_basis = torch.stack(selected_hidden)              # [C, K, H]
    basis_eigvals = torch.stack(selected_vals)               # [C, K]
    basis_vectors = project_eigenvectors_to_input(
        hidden_basis.transpose(1, 2),
        model.embedding_weight,
    )                                                        # [C, K, D]
    if normalise_eigenvectors:
        basis_vectors = _normalise_rows(basis_vectors)

    masks_raw: list[Tensor] = []
    masks_eig: list[Tensor] = []
    target_labels: list[int] = []
    ranks: list[int] = []
    selected_eigvals: list[Tensor] = []

    for c_t in classes.tolist():
        frame = basis_vectors[c_t]                           # [K, D]
        pinv_rows = torch.linalg.pinv(frame.T)                # [K, D]
        for rank_idx in range(attack_ranks):
            masks_raw.append(pinv_rows[rank_idx])
            masks_eig.append(frame[rank_idx])
            target_labels.append(int(c_t))
            ranks.append(rank_idx + 1)
            selected_eigvals.append(basis_eigvals[c_t, rank_idx])

    masks_pinv_raw = torch.stack(masks_raw)
    masks_pinv = _normalise_rows(masks_pinv_raw) if normalise_masks else masks_pinv_raw
    masks_eigvec = _normalise_rows(torch.stack(masks_eig))

    return AdversarialMasks(
        masks_pinv=masks_pinv,
        masks_pinv_raw=masks_pinv_raw,
        masks_eigvec=masks_eigvec,
        target_classes=torch.tensor(target_labels, device=eigvals.device, dtype=torch.long),
        eigen_ranks=torch.tensor(ranks, device=eigvals.device, dtype=torch.long),
        eigvals_selected=torch.stack(selected_eigvals),
        basis_vectors=basis_vectors,
        basis_eigvals=basis_eigvals,
    )


def compute_permuted_masks(
    masks: Tensor,
    seed: int = 42,
) -> Tensor:
    """Randomly permute each mask's pixels, matching the Figure 7 baseline."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    rows = []
    masks_cpu = masks.detach().cpu()
    for row in masks_cpu:
        rows.append(row[torch.randperm(row.numel(), generator=generator)])
    return torch.stack(rows).to(device=masks.device, dtype=masks.dtype)


def compute_random_masks(
    num_masks: int,
    d_input: int,
    seed: int = 42,
    device: torch.device | str = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Unit-L2 Gaussian masks.

    Kept as a diagnostic baseline; Figure 7's paper baseline is
    ``compute_permuted_masks``.
    """
    generator = torch.Generator(device='cpu').manual_seed(seed)
    raw = torch.randn(num_masks, d_input, generator=generator, dtype=dtype)
    raw = _normalise_rows(raw)
    return raw.to(device)


def apply_spatial_mask(
    masks: Tensor,
    keep_mask: Tensor,
    *,
    normalise: bool = True,
) -> Tensor:
    """Zero out disallowed input dimensions, optionally restoring unit norm."""
    keep = keep_mask.to(device=masks.device, dtype=masks.dtype).view(1, -1)
    if keep.shape[1] != masks.shape[1]:
        raise ValueError(
            f'keep_mask has {keep.shape[1]} elements but masks have dimension {masks.shape[1]}.'
        )
    masked = masks * keep
    return _normalise_rows(masked) if normalise else masked


@torch.no_grad()
def compute_low_activity_mask(
    loader,
    *,
    d_input: int,
    device: torch.device,
    active_threshold: float = 0.01,
    pixel_threshold: float = 0.0,
    max_batches: int | None = None,
) -> Tensor:
    """Pixels active on less than ``active_threshold`` of samples.

    This reproduces the paper's edge-only variant for unregularised MNIST
    models.  For normalised RGB datasets, choose ``pixel_threshold`` in that
    transformed input space.
    """
    counts = torch.zeros(d_input, device=device)
    total = 0
    for batch_idx, (x, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True).flatten(start_dim=1)
        if x.shape[1] != d_input:
            raise ValueError(f'Expected flattened input dimension {d_input}, got {x.shape[1]}.')
        counts += (x > pixel_threshold).float().sum(dim=0)
        total += x.shape[0]
        if max_batches is not None and batch_idx >= max_batches:
            break
    if total == 0:
        raise ValueError('Cannot compute low-activity mask from an empty loader.')
    return (counts / total) < active_threshold


@torch.no_grad()
def evaluate_attacks(
    model: nn.Module,
    loader,
    masks: Tensor,
    target_classes: Tensor,
    magnitudes: Iterable[float],
    device: torch.device,
    clip_range: tuple[float, float] | None = None,
    input_shape: tuple[int, int, int] | None = None,
    max_batches: int | None = None,
) -> AttackEvaluation:
    """Sweep targeted masks and report accuracy plus target-hit rate."""
    model.eval()
    masks = masks.to(device)
    targets = target_classes.to(device=device, dtype=torch.long)
    if masks.shape[0] != targets.numel():
        raise ValueError('masks and target_classes must have the same first dimension.')

    magnitudes_t = torch.as_tensor(list(magnitudes), dtype=masks.dtype, device=device)
    if magnitudes_t.numel() == 0:
        raise ValueError('magnitudes must contain at least one value.')

    num_masks, d_input = masks.shape
    num_magnitudes = magnitudes_t.numel()
    target_hits = torch.zeros(num_magnitudes, num_masks, device=device)
    target_counts = torch.zeros(num_magnitudes, num_masks, device=device)
    correct = torch.zeros(num_magnitudes, num_masks, device=device)
    counts = torch.zeros(num_magnitudes, num_masks, device=device)

    clean_correct = 0
    total_examples = 0

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch = x.size(0)
        x_flat = x.flatten(start_dim=1)
        if x_flat.shape[1] != d_input:
            raise ValueError(
                f'Mask dimension ({d_input}) does not match flattened input '
                f'dimension ({x_flat.shape[1]}).'
            )

        clean_logits = model(x)
        clean_correct += (clean_logits.argmax(dim=-1) == y).sum().item()
        total_examples += batch

        additive = magnitudes_t.view(num_magnitudes, 1, 1, 1) * masks.view(1, num_masks, 1, d_input)
        x_adv_flat = x_flat.view(1, 1, batch, d_input) + additive
        if clip_range is not None:
            x_adv_flat = x_adv_flat.clamp(*clip_range)

        if input_shape is None:
            x_adv = x_adv_flat.reshape(num_magnitudes * num_masks * batch, d_input)
        else:
            x_adv = x_adv_flat.reshape(num_magnitudes * num_masks * batch, *input_shape)
        pred = model(x_adv).argmax(dim=-1).view(num_magnitudes, num_masks, batch)

        target_view = targets.view(1, num_masks, 1)
        eligible = y.view(1, 1, batch) != target_view
        target_hits += ((pred == target_view) & eligible).sum(dim=2).float()
        target_counts += eligible.expand(num_magnitudes, -1, -1).sum(dim=2).float()

        correct += (pred == y.view(1, 1, batch)).sum(dim=2).float()
        counts += float(batch)

        if max_batches is not None and batch_idx >= max_batches:
            break

    target_success_by_mask = (target_hits / target_counts.clamp_min(1.0)).cpu()
    accuracy_by_mask = (correct / counts.clamp_min(1.0)).cpu()

    return AttackEvaluation(
        magnitudes=magnitudes_t.cpu(),
        target_success_by_mask=target_success_by_mask,
        accuracy_by_mask=accuracy_by_mask,
        target_success_mean=target_success_by_mask.mean(dim=1),
        accuracy_mean=accuracy_by_mask.mean(dim=1),
        clean_acc=clean_correct / max(total_examples, 1),
        num_examples=total_examples,
    )
