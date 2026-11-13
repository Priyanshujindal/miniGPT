
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import GPTEmbeddings
from model.transformer_block import TransformerBlock
from model.normalization import LayerNorm
from model.layers import Linear


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        num_heads=4,
        num_layers=4,
        max_sequence_length=128,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.num_layers = num_layers

        # Token and positional embeddings.
        self.embeddings = GPTEmbeddings(
            vocab_size,
            embedding_dim,
            max_sequence_length,
            dropout,
        )

        # Transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim,
                num_heads,
                dropout,
            )
            for _ in range(num_layers)
        ])

        # Final normalization.
        self.final_norm = LayerNorm(embedding_dim)

        # Output projection shares token embedding weights.
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):

        if isinstance(module, Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):

        if input_ids.ndim != 2:
            raise ValueError(
                "Input must have shape [batch, sequence]."
            )

        if input_ids.size(1) > self.max_sequence_length:
            raise ValueError(
                "Sequence exceeds model context length."
            )

        if input_ids.numel() > 0:

            if (
                input_ids.min() < 0
                or input_ids.max() >= self.vocab_size
            ):
                raise ValueError(
                    "Input contains an invalid token ID."
                )

        # Embedding layer.
        x = self.embeddings(input_ids)

        # Transformer layers.
        for block in self.blocks:
            x = block(x)

        # Final normalization.
        x = self.final_norm(x)

        # Weight tying:
        # Reuse token embedding weights as output projection.
        logits = F.linear(
            x,
            self.embeddings.token_embedding.weight,
        )

        return logits
