__all__ = ["MaskedAutoEncoder"]
from argparse import Namespace
from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from deep_learning_starter.modules import Transformer
from deep_learning_starter.modules.rmsnorm import RMSNorm


class MaskedAutoEncoder(nn.Module):

    def __init__(
        self,
        mask_ratio: float,
        patch_size: int,
        hidden_size: int,
        head_size: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.eps = eps
        self.patch_conv = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size, device=device, dtype=dtype)
        self.encoder = Transformer(
            hidden_size=hidden_size, head_size=head_size, num_heads=num_heads, num_layers=num_encoder_layers, device=device, dtype=dtype
        )
        self.decoder = Transformer(
            hidden_size=hidden_size, head_size=head_size, num_heads=num_heads, num_layers=num_decoder_layers, device=device, dtype=dtype
        )
        self.patch_norm = RMSNorm(hidden_size, eps=eps, device=device, dtype=dtype)
        self.nonlinear = nn.SiLU(True)
        self.patch_deconv = nn.ConvTranspose2d(hidden_size, 3, kernel_size=patch_size, stride=patch_size, device=device, dtype=dtype)

    def forward(self, input: Tensor, target: Optional[Tensor] = None) -> Namespace:

        patch_embeds = self.patch_conv(input)
        hidden_states = patch_embeds.flatten(-2).transpose(-1, -2)
        latent_states = torch.zeros_like(hidden_states)
        sample_index = torch.arange(hidden_states.shape[1], device=input.get_device())
        if self.training:
            sample_index = torch.randperm(hidden_states.shape[1], device=input.get_device())[
                : int(hidden_states.shape[1] * (1 - self.mask_ratio))
            ]
            hidden_states = hidden_states[:, sample_index]
        hidden_states = self.encoder(hidden_states)

        latent_states = latent_states.index_copy_(1, sample_index, hidden_states)

        last_hidden_states = self.decoder(latent_states)
        logits: Tensor = self.patch_norm(last_hidden_states)
        logits = self.nonlinear(logits)
        logits = (
            logits.index_copy_(1, sample_index, logits.index_select(1, sample_index).clone().detach())
            .transpose(-1, -2)
            .view_as(patch_embeds)
        )
        logits = self.patch_deconv(logits)
        loss = None
        if target is not None:
            loss = F.mse_loss(logits, target)

        return Namespace(latent_states=latent_states, last_hidden_states=last_hidden_states, logits=logits, loss=loss)
