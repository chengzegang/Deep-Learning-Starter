__all__ = [
    "Attention",
    "TransformerLayer",
    "ConditionalTransformerLayer",
    "Transformer",
    "ConditionalTransformer",
]
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Optional, Tuple
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .rotary import RotaryEmbedding


def _naive_scaled_dot_product_flash_attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
    return attn_weight @ V


class Attention(nn.Module):
    def __init__(
        self,
        hidden_states: int,
        num_heads: int,
        head_size: int,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Attention module that performs scaled dot-product attention.

        Args:
            hidden_states (int): The number of hidden states.
            num_heads (int): The number of attention heads.
            head_size (int): The size of each attention head.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 8192.
            freq_base (int, optional): The base frequency for the rotary embeddings. Defaults to 10000.
            dtype (torch.dtype, optional): The data type of the module's parameters. Defaults to None.
            device (torch.device, optional): The device where the module's parameters are stored. Defaults to None.
        """
        super().__init__()
        self.hidden_states = hidden_states
        self.num_heads = num_heads
        self.head_size = head_size

        self.q_proj = nn.Linear(
            hidden_states,
            num_heads * head_size,
            dtype=dtype,
            device=device,
        )
        self.k_proj = nn.Linear(
            hidden_states,
            num_heads * head_size,
            dtype=dtype,
            device=device,
        )
        self.v_proj = nn.Linear(
            hidden_states,
            num_heads * head_size,
            dtype=dtype,
            device=device,
        )
        self.out_proj = nn.Linear(
            num_heads * head_size,
            hidden_states,
            dtype=dtype,
            device=device,
        )
        self.rotary = RotaryEmbedding(
            head_size,
            max_seq_length,
            freq_base,
            dtype=dtype,
            device=device,
        )

    def forward(
        self, query_embeds: Tensor, keyvalue_embeds: Tensor, attention_mask: Optional[Tensor] = None, is_causal: bool = False
    ) -> Tensor:
        """
        Forward pass of the Attention module.

        Args:
            input_embeds (Tensor): The input embeddings.
            attention_mask (Tensor, optional): The attention mask. Defaults to None.
            is_causal (bool, optional): Whether the attention is causal or not. Defaults to False.

        Returns:
            Tensor: The output of the Attention module.
        """
        q = self.q_proj(query_embeds)
        k = self.k_proj(keyvalue_embeds)
        v = self.v_proj(keyvalue_embeds)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_size).transpose(-2, -3)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_size).transpose(-2, -3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_size).transpose(-2, -3)
        q, k = self.rotary(q, k)

        # attn_weights = _naive_scaled_dot_product_flash_attention(q, k, v) for export
        attn_weights = F.scaled_dot_product_attention(q, k, v, attention_mask, is_causal=is_causal)

        attn_weights = attn_weights.transpose(-2, -3).flatten(-2)

        out = self.out_proj(attn_weights)

        return out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_size: int,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        eps: float = 1e-5,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initializes a TransformerLayer module.

        Args:
            hidden_size (int): The size of the hidden state.
            num_heads (int): The number of attention heads.
            head_size (int): The size of each attention head.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 8192.
            freq_base (int, optional): The base frequency for sinusoidal positional encoding. Defaults to 10000.
            eps (float, optional): The epsilon value for RMSNorm. Defaults to 1e-5.
            dtype (torch.dtype, optional): The data type of the tensors. Defaults to None.
            device (torch.device, optional): The device to use for computation. Defaults to None.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.ln1 = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype, device=device)
        self.self_attn = Attention(
            hidden_states=hidden_size,
            num_heads=num_heads,
            head_size=head_size,
            max_seq_length=max_seq_length,
            freq_base=freq_base,
            dtype=dtype,
            device=device,
        )

        self.ln2 = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype, device=device)
        self.mlp = SwiGLU(
            in_features=hidden_size,
            hidden_features=hidden_size * 8 // 3,
            out_features=hidden_size,
            dtype=dtype,
            device=device,
        )

    def forward(self, input_embeds: Tensor, attention_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        """
        Performs a forward pass through the TransformerLayer module.

        Args:
            input_embeds (Tensor): The input embeddings.
            attention_mask (Tensor, optional): The attention mask. Defaults to None.
            is_causal (bool, optional): Whether the attention is causal or not. Defaults to False.

        Returns:
            Tensor: The output embeddings.
        """
        residual = input_embeds
        input_embeds = self.ln1(input_embeds)
        input_embeds = self.self_attn(input_embeds, input_embeds, attention_mask, is_causal)
        input_embeds = input_embeds + residual

        residual = input_embeds
        input_embeds = self.ln2(input_embeds)
        input_embeds = self.mlp(input_embeds)
        input_embeds = input_embeds + residual

        return input_embeds


class ConditionalTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_size: int,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        eps: float = 1e-5,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Conditional Decoder Layer of a Transformer model.

        Args:
            hidden_size (int): The size of the hidden state.
            num_heads (int): The number of attention heads.
            head_size (int): The size of each attention head.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 8192.
            freq_base (int, optional): The base frequency for positional encoding. Defaults to 10000.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-5.
            dtype (torch.dtype, optional): The data type of the tensors. Defaults to None.
            device (torch.device, optional): The device to use for computation. Defaults to None.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.ln1 = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype, device=device)
        self.self_attn = Attention(
            hidden_states=hidden_size,
            num_heads=num_heads,
            head_size=head_size,
            max_seq_length=max_seq_length,
            freq_base=freq_base,
            dtype=dtype,
            device=device,
        )

        self.ln2 = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype, device=device)
        self.cross_attn = Attention(
            hidden_states=hidden_size,
            num_heads=num_heads,
            head_size=head_size,
            max_seq_length=max_seq_length,
            freq_base=freq_base,
            dtype=dtype,
            device=device,
        )

        self.ln3 = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype, device=device)
        self.mlp = SwiGLU(
            in_features=hidden_size,
            hidden_features=hidden_size * 8 // 3,
            out_features=hidden_size,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        input_embeds: Tensor,
        cond_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the Conditional Decoder Layer.

        Args:
            input_embeds (Tensor): The input embeddings.
            cond_embeds (Tensor): The conditional embeddings.
            attention_mask (Tensor, optional): The attention mask. Defaults to None.

        Returns:
            Tensor: The output embeddings.
        """
        residual = input_embeds
        input_embeds = self.ln1(input_embeds)
        input_embeds = self.self_attn(input_embeds, input_embeds, attention_mask)
        input_embeds = input_embeds + residual

        residual = input_embeds
        input_embeds = self.ln2(input_embeds)
        input_embeds = self.cross_attn(input_embeds, cond_embeds, attention_mask)
        input_embeds = input_embeds + residual

        residual = input_embeds
        input_embeds = self.ln3(input_embeds)
        input_embeds = self.mlp(input_embeds)
        input_embeds = input_embeds + residual

        return input_embeds


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        eps: float = 1e-5,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Transformer module that applies multiple layers of TransformerLayer to the input embeddings.

        Args:
            hidden_size (int): The size of the hidden state.
            num_layers (int): The number of layers in the Transformer.
            num_heads (int): The number of attention heads.
            head_size (int): The size of each attention head.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 8192.
            freq_base (int, optional): The base frequency for sinusoidal positional encoding. Defaults to 10000.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-5.
            dtype (torch.dtype, optional): The data type of the input tensors. Defaults to None.
            device (torch.device, optional): The device to use for computation. Defaults to None.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    head_size=head_size,
                    max_seq_length=max_seq_length,
                    freq_base=freq_base,
                    eps=eps,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input_embeds: Tensor, attention_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        """
        Forward pass of the Transformer module.

        Args:
            input_embeds (Tensor): The input embeddings.
            attention_mask (Tensor, optional): The attention mask. Defaults to None.
            is_causal (bool, optional): Whether the attention is causal or not. Defaults to False.

        Returns:
            Tensor: The output embeddings after applying multiple layers of TransformerLayer.
        """
        for layer in self.layers:
            input_embeds = layer(input_embeds, attention_mask, is_causal)

        return input_embeds


class ConditionalTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        eps: float = 1e-5,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initializes a ConditionalTransformer module.

        Args:
            hidden_size (int): The size of the hidden state.
            num_layers (int): The number of layers in the transformer.
            num_heads (int): The number of attention heads.
            head_size (int): The size of each attention head.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 8192.
            freq_base (int, optional): The base frequency for positional encoding. Defaults to 10000.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-5.
            dtype (torch.dtype, optional): The data type of the tensors. Defaults to None.
            device (torch.device, optional): The device to use for computation. Defaults to None.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size

        self.layers = nn.ModuleList(
            [
                ConditionalTransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    head_size=head_size,
                    max_seq_length=max_seq_length,
                    freq_base=freq_base,
                    eps=eps,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        input_embeds: Tensor,
        cond_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the ConditionalTransformer module.

        Args:
            input_embeds (Tensor): The input embeddings.
            cond_embeds (Tensor): The conditional embeddings.
            attention_mask (Tensor, optional): The attention mask. Defaults to None.

        Returns:
            Tensor: The output embeddings.
        """
        for layer in self.layers:
            input_embeds = layer(input_embeds, cond_embeds, attention_mask)

        return input_embeds
