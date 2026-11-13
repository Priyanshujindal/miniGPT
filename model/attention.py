
import math
import torch
import torch.nn as nn

from model.layers import Linear



import math
import torch
import torch.nn as nn

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

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        # Causal mask is registered as a buffer.
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    1,
                    1,
                    1024,
                    1024,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

    def forward(self, x):

        batch_size, sequence_length, channels = x.shape

        if sequence_length > self.causal_mask.size(-1):
            raise ValueError("Sequence length exceeds attention mask.")

        qkv = self.qkv(x)

        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(dim=0)

        scores = query @ key.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[
            :, :,
            :sequence_length,
            :sequence_length,
        ]

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        output = attention_weights @ value

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            channels,
        )

        output = self.output_projection(output)

        return self.residual_dropout(output)
