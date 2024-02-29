__all__ = [
    "SwiGLU",
]
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional


@torch.jit.script
def fused_swiglu(x: Tensor, w1: Tensor, b1: Optional[Tensor], w2: Tensor, b2: Optional[Tensor], w3: Tensor, b3: Optional[Tensor]) -> Tensor:
    x1 = F.linear(x, w1, b1)
    x2 = F.linear(x, w2, b2)
    hidden = F.silu(x1) * x2
    return F.linear(hidden, w3, b3)


class SwiGLU(nn.Module):
    """
    SwiGLU module that applies the SwiGLU activation function to the input.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        out_features (int, optional): Number of output features. Defaults to None, which sets it equal to in_features.
        dtype (torch.dtype, optional): Data type of the weights and biases. Defaults to None.
        device (torch.device, optional): Device where the weights and biases are stored. Defaults to None.

    Attributes:
        w1 (nn.Linear): Linear layer for the first transformation.
        w2 (nn.Linear): Linear layer for the second transformation.
        w3 (nn.Linear): Linear layer for the third transformation.
        hidden_features (int): Number of hidden features.
        out_features (int): Number of output features.
        in_features (int): Number of input features.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features

        self.w1 = nn.Linear(
            in_features,
            hidden_features,
            dtype=dtype,
            device=device,
            bias=False,
        )
        self.w2 = nn.Linear(
            in_features,
            hidden_features,
            dtype=dtype,
            device=device,
            bias=False,
        )
        self.w3 = nn.Linear(
            hidden_features,
            out_features,
            dtype=dtype,
            device=device,
            bias=False,
        )
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SwiGLU module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the SwiGLU activation function.
        """
        return fused_swiglu(x, self.w1.weight, self.w1.bias, self.w2.weight, self.w2.bias, self.w3.weight, self.w3.bias)
