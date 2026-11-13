
import math
import torch
import torch.nn as nn

from model.layers import Linear


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):

        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.head_dim = embedding_dim // num_heads

        # Generate Query, Key and Value together.
        self.qkv = Linear(
            embedding_dim,
            3 * embedding_dim
        )

        self.output_projection = Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        batch_size, sequence_length, channels = x.shape

        # Generate Q, K and V.
        qkv = self.qkv(x)

        # [B, T, 3C] -> [B, T, 3, H, D]
        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim
        )

        # [3, B, H, T, D]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(dim=0)

        # Calculate attention scores.
        scores = query @ key.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        # Create lower-triangular causal mask.
        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device
            )
        )

        # Prevent attention to future positions.
        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf")
        )

        # Convert scores into probabilities.
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # Apply attention to Value vectors.
        output = attention_weights @ value

        # [B, H, T, D] -> [B, T, C]
        output = output.transpose(
            1,
            2
        ).contiguous().view(
            batch_size,
            sequence_length,
            channels
        )

        output = self.output_projection(output)

        return output
