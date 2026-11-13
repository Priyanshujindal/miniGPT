
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):

        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Learnable embedding matrix.
        self.weight = nn.Parameter(
            torch.randn(
                vocab_size,
                embedding_dim
            ) * 0.02
        )

    def forward(self, token_ids):

        # token_ids shape: [B, T]
        # weight shape: [V, C]

        embeddings = self.weight[token_ids]

        # Output shape: [B, T, C]

        return embeddings


class PositionalEmbedding(nn.Module):

    def __init__(self, max_sequence_length, embedding_dim):

        super().__init__()

        self.max_sequence_length = max_sequence_length

        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.randn(
                max_sequence_length,
                embedding_dim
            ) * 0.02
        )

    def forward(self, token_ids):

        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                "Sequence length exceeds maximum position limit."
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        position_embeddings = self.weight[positions]

        # [T, C] -> [1, T, C]
        position_embeddings = (
            position_embeddings.unsqueeze(0)
        )

        return position_embeddings.expand(
            batch_size,
            -1,
            -1
        )



class GPTEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        max_sequence_length,
        dropout=0.1,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim,
        )

        self.position_embedding = PositionalEmbedding(
            max_sequence_length,
            embedding_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        embeddings = (
            self.token_embedding(token_ids)
            + self.position_embedding(token_ids)
        )

        return self.dropout(embeddings)

