
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

        self.norm1 = LayerNorm(embedding_dim)

        self.attention = CausalSelfAttention(
            embedding_dim,
            num_heads,
            dropout,
        )

        self.norm2 = LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim,
            dropout,
        )

    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x
