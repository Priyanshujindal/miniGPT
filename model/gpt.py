
import torch
import torch.nn as nn

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
        max_sequence_length=128
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

        # Token + positional embeddings.
        self.embeddings = GPTEmbeddings(
            vocab_size,
            embedding_dim,
            max_sequence_length
        )

        # Stack Transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim,
                num_heads
            )
            for _ in range(num_layers)
        ])

        # Final normalization.
        self.final_norm = LayerNorm(
            embedding_dim
        )

        # Project hidden states to vocabulary logits.
        self.output_head = Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

    def forward(self, input_ids):

        if input_ids.ndim != 2:
            raise ValueError(
                "Input must have shape [batch, sequence]."
            )

        if input_ids.size(1) > self.max_sequence_length:
            raise ValueError(
                "Sequence exceeds model context length."
            )

        if input_ids.min() < 0 or input_ids.max() >= self.vocab_size:
            raise ValueError(
                "Input contains an invalid token ID."
            )

        # Embedding lookup.
        x = self.embeddings(input_ids)

        # Transformer processing.
        for block in self.blocks:

            x = block(x)

        # Final normalization.
        x = self.final_norm(x)

        # Vocabulary predictions.
        logits = self.output_head(x)

        return logits
