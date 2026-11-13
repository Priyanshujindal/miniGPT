
import math
import torch
import torch.nn as nn


class Linear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True
    ):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Learnable weights.
        self.weight = nn.Parameter(
            torch.empty(
                in_features,
                out_features
            )
        )

        if bias:

            self.bias = nn.Parameter(
                torch.zeros(out_features)
            )

        else:

            self.register_parameter(
                "bias",
                None
            )

        self.reset_parameters()

    def reset_parameters(self):

        bound = 1 / math.sqrt(self.in_features)

        nn.init.uniform_(
            self.weight,
            -bound,
            bound
        )

    def forward(self, x):

        output = x @ self.weight

        if self.bias is not None:

            output = output + self.bias

        return output


class GELU(nn.Module):

    def forward(self, x):

        return 0.5 * x * (
            1.0 + torch.erf(
                x / math.sqrt(2.0)
            )
        )
