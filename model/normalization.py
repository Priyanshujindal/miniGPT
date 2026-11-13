
import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(
        self,
        embedding_dim,
        eps=1e-5
    ):

        super().__init__()

        self.eps = eps

        # Learnable scaling parameter.
        self.gamma = nn.Parameter(
            torch.ones(embedding_dim)
        )

        # Learnable shifting parameter.
        self.beta = nn.Parameter(
            torch.zeros(embedding_dim)
        )

    def forward(self, x):

        # Calculate mean over the last dimension.
        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        # Population variance.
        variance = (
            (x - mean) ** 2
        ).mean(
            dim=-1,
            keepdim=True
        )

        # Normalize.
        normalized = (
            (x - mean)
            / torch.sqrt(variance + self.eps)
        )

        # Learnable scale and shift.
        output = (
            self.gamma * normalized
            + self.beta
        )

        return output
