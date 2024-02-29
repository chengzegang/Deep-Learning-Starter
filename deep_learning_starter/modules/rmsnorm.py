__all__ = [
    "RMSNorm",
    "SpatialRMSNorm",
]
import torch
from torch import nn, Tensor
from typing import Optional


@torch.jit.script
def fused_rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-8) -> Tensor:
    x = x * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + eps) * weight
    return x


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-8,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        RMSNorm module applies root mean square normalization to the input tensor.

        Args:
            hidden_size (int): The size of the hidden dimension.
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-8.
            dtype (torch.dtype, optional): The desired data type of the weight tensor. Defaults to None.
            device (torch.device, optional): The desired device of the weight tensor. Defaults to None.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the RMSNorm module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying RMSNorm.
        """
        return fused_rmsnorm(x, self.weight, self.eps)  # eps of bfloat16


@torch.jit.script
def fused_spatial_rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-8) -> Tensor:
    shape = x.shape
    x = x.view(x.shape[0], x.shape[1], -1)
    x = x * torch.rsqrt((x**2).mean(dim=1, keepdim=True) + eps) * weight.view(-1, 1)
    x = x.view(shape)
    return x


class SpatialRMSNorm(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-8,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Spatial Root Mean Square Normalization (RMSNorm) module.

        Args:
            num_features (int): Number of input features.
            eps (float, optional): Small value added to the denominator for numerical stability. Default is 1e-8.
            dtype (torch.dtype, optional): Data type of the parameters. Default is None.
            device (torch.device, optional): Device where the parameters are stored. Default is None.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features, dtype=dtype, device=device))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Forward pass of the SpatialRMSNorm module.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor after applying SpatialRMSNorm.
        """
        return fused_spatial_rmsnorm(hidden_states, self.scale, self.eps)
