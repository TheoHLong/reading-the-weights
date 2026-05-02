from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import einsum
from torch import Tensor

from src.model import BilinearImageClassifier


# Gates compatible with the paper's bilinear-style weight decomposition.
# - None: pure bilinear MLP (Pearce et al. 2024).
# - 'sqs': Signed Quadratic Shrink (Abohwo & Mosen 2025). The paper applies the
#   same Pearce-style decomposition directly to the SQS-GLU weights, ignoring
#   the σ non-linearity ("first-order linearization"). This is justified
#   empirically by SQS being quasi-linear for |x| << 1 and is the procedure
#   used to produce Figure 2 in the SQS paper.
DECOMPOSABLE_GATES: frozenset[str | None] = frozenset({None, 'sqs'})


@dataclass
class DecompositionArtifacts:
    bilinear_tensor: Tensor
    symmetrized_tensor: Tensor
    eigenvalues: Tensor
    eigenvectors_hidden: Tensor
    eigenvectors_input: Tensor

    def to_payload(self) -> dict[str, Tensor]:
        return {
            'bilinear_tensor': self.bilinear_tensor.cpu(),
            'symmetrized_tensor': self.symmetrized_tensor.cpu(),
            'eigenvalues': self.eigenvalues.cpu(),
            'eigenvectors_hidden': self.eigenvectors_hidden.cpu(),
            'eigenvectors_input': self.eigenvectors_input.cpu(),
        }


def build_bilinear_tensor(model: BilinearImageClassifier) -> Tensor:
    gate = getattr(model, 'gate', None)
    if gate not in DECOMPOSABLE_GATES:
        raise ValueError(
            'Weight-space bilinear decomposition supports gate in '
            f'{sorted(repr(g) for g in DECOMPOSABLE_GATES)}, got gate={gate!r}. '
            "For SQS we follow the paper's first-order procedure: form A=W^T V "
            'directly from the SQS-GLU weights, treating σ as identity. Gates '
            'like ReLU/GELU/SiLU break this approximation and are not supported.'
        )

    if len(model.blocks) != 1:
        raise ValueError('Task A decomposition supports single-layer models only.')

    left, right = model.bilinear_weights[0].unbind(0)
    return einsum(
        model.output_weight,
        left,
        right,
        'cls out, out in_left, out in_right -> cls in_left in_right',
    )


def symmetrize_bilinear_tensor(bilinear_tensor: Tensor) -> Tensor:
    return 0.5 * (bilinear_tensor + bilinear_tensor.mT)


def project_eigenvectors_to_input(eigenvectors_hidden: Tensor, embedding_weight: Tensor) -> Tensor:
    return einsum(
        eigenvectors_hidden,
        embedding_weight,
        'cls hidden eig, hidden inp -> cls eig inp',
    )


def spectral_effective_rank(eigenvalues: Tensor, eps: float = 1e-12) -> Tensor:
    magnitudes = eigenvalues.abs()
    weights = magnitudes / magnitudes.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=-1)
    return entropy.exp()


@torch.no_grad()
def decompose_bilinear_model(model: BilinearImageClassifier) -> DecompositionArtifacts:
    bilinear_tensor = build_bilinear_tensor(model)
    symmetrized_tensor = symmetrize_bilinear_tensor(bilinear_tensor)
    eigenvalues, eigenvectors_hidden = torch.linalg.eigh(symmetrized_tensor)
    eigenvectors_input = project_eigenvectors_to_input(eigenvectors_hidden, model.embedding_weight)

    return DecompositionArtifacts(
        bilinear_tensor=bilinear_tensor,
        symmetrized_tensor=symmetrized_tensor,
        eigenvalues=eigenvalues,
        eigenvectors_hidden=eigenvectors_hidden,
        eigenvectors_input=eigenvectors_input,
    )
