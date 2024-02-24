__all__ = [
    "RotaryEmbedding",
]
import torch
from torch import nn, Tensor
from typing import Optional, Tuple


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # NOTE: This could probably be moved to Triton

    # Handle a possible sequence length mismatch in between q and k
    cos = cos[..., : x.shape[-2], :]
    sin = sin[..., : x.shape[-2], :]

    return (x * cos.type_as(x)) + (rotate_half(x) * sin.type_as(x))


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Rotary Embedding module for Transformer-based models.

        Args:
            dim_model (int): The dimensionality of the model.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 8192.
            freq_base (int, optional): The base frequency for the rotary embedding. Defaults to 10000.
            dtype (torch.dtype, optional): The data type of the rotary embedding. Defaults to None.
            device (torch.device, optional): The device to use for the rotary embedding. Defaults to None.
        """
        super().__init__()
        self.dim_model = dim_model
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (
            freq_base
            ** (torch.arange(0, dim_model, 2, dtype=torch.float32, device=device, requires_grad=False) / dim_model)
        ).to(dtype=dtype)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        _cos_cached, s_sin_cached = self._update_cos_sin_tables(max_seq_length, torch.float32, device)
        self.register_buffer("_cos_cached", _cos_cached.to(dtype), persistent=False)
        self.register_buffer("_sin_cached", s_sin_cached.to(dtype), persistent=False)

    def _update_cos_sin_tables(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> Tuple[Tensor, Tensor]:
        """
        Update the cosine and sine tables for rotary embedding.

        Args:
            seq_len (int): The sequence length.

        Returns:
            Tuple[Tensor, Tensor]: The cosine and sine tables.
        """
        t = torch.arange(seq_len, dtype=dtype, device=device, requires_grad=False)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        _cos_cached = emb.cos()[None, None, :, :]
        _sin_cached = emb.sin()[None, None, :, :]

        return _cos_cached, _sin_cached

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary positional embedding to the input tensors.

        Args:
            q (Tensor): The query tensor.
            k (Tensor): The key tensor.

        Returns:
            Tuple[Tensor, Tensor]: The query tensor with rotary positional embedding,
                and the key tensor with rotary positional embedding.
        """
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )
