
import torch.nn as nn

from model.attention import CausalSelfAttention
from model.feed_forward import FeedForward
from model.normalization import LayerNorm


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=0.1,
    ):
        super().__init__()

        # First Pre-LayerNorm.
        self.norm1 = LayerNorm(embedding_dim)

        # Optimized causal self-attention.
        self.attention = CausalSelfAttention(
            embedding_dim,
            num_heads,
            dropout,
        )

        # Second Pre-LayerNorm.
        self.norm2 = LayerNorm(embedding_dim)

        # Feed-forward network.
        self.feed_forward = FeedForward(
            embedding_dim,
            dropout,
        )

    def forward(self, x):

        # Attention residual connection.
        x = x + self.attention(
            self.norm1(x)
        )

        # Feed-forward residual connection.
        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x
