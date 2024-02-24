__all__ = ["VocabularizedPositionalEmbedding", "SinePositionalEmbedding"]
import torch
from torch import nn, Tensor
import math


class VocabularizedPositionalEmbedding(nn.Module):
    def __init__(self, dim_model: int, max_seq_len: int = 10000):
        super().__init__()
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.register_buffer("positional_ids", torch.arange(max_seq_len), persistent=False)
        self.positional_embedding = nn.Embedding(max_seq_len, dim_model)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        pos = self.positional_embedding(self.positional_ids[:seq_len])
        return x + pos.unsqueeze(0)


class SinePositionalEmbedding(nn.Module):
    def __init__(self, dim_model: int, freq_base: int = 10000):
        super().__init__()
        self.dim_model = dim_model
        self.freq_base = freq_base

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        pos = torch.arange(0, seq_len, device=x.device, dtype=torch.float32).unsqueeze(1).repeat(1, self.dim_model)
        dim = torch.arange(0, self.dim_model, device=x.device, dtype=torch.float32).unsqueeze(0).repeat(seq_len, 1)
        div = torch.exp(-math.log(self.freq_base) * (2 * (dim // 2) / self.dim_model))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        output = x.unsqueeze(-1) if x.ndim == 2 else x

        return output + pos.unsqueeze(0)
