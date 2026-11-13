
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Linear


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=0.1,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by number of heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv = Linear(
            embedding_dim,
            3 * embedding_dim,
        )

        self.output_projection = Linear(
            embedding_dim,
            embedding_dim,
        )

        # SDPA applies attention dropout internally.
        self.attention_dropout = dropout

        # Dropout after the attention output projection.
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):

        batch_size, sequence_length, channels = x.shape

        # Project input into query, key and value.
        qkv = self.qkv(x)

        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )

        # Shape: [3, batch, heads, sequence, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(dim=0)

        # Optimized scaled dot-product attention.
        # PyTorch automatically applies causal masking.
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=(
                self.attention_dropout
                if self.training
                else 0.0
            ),
            is_causal=True,
        )

        # [batch, heads, sequence, head_dim]
        # -> [batch, sequence, embedding_dim]
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            channels,
        )

        output = self.output_projection(output)

        return self.residual_dropout(output)
