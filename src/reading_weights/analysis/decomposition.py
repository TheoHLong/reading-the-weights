from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import einsum
from torch import Tensor

from reading_weights.models.image_classifier import BilinearImageClassifier


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
