
import torch.nn as nn

from model.layers import Linear, GELU


class FeedForward(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.network = nn.Sequential(
            Linear(
                embedding_dim,
                4 * embedding_dim
            ),

            GELU(),

            Linear(
                4 * embedding_dim,
                embedding_dim
            )
        )

    def forward(self, x):

        return self.network(x)
