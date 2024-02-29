__all__ = [
    "VectorQuantizedVariationalAutoEncoderOutput",
    "VectorQuantization",
    "VectorQuantizedVariationalAutoEncoder",
    "VectorQuantizedVariationalAutoEncoder2d",
    "VectorQuantizedVariationalAutoEncoder3d",
    "VQVAE",
    "VQVAE2d",
    "VQVAE3d",
]
import math
from typing import Optional
from deep_learning_starter.modules import UnetEncoder2d, UnetDecoder2d, UnetEncoder3d, UnetDecoder3d
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from . import _utils
from kornia.color import rgb_to_lab
from dataclasses import dataclass


@dataclass
class VectorQuantizedVariationalAutoEncoderOutput:
    sample: Tensor
    target: Optional[Tensor]
    latent_dist: "VectorQuantization"
    rec_loss: Tensor
    kl_loss: Tensor
    loss: Tensor

    @property
    def plot(self):
        if self.target is not None:
            return _utils.plot_pair(self.sample[0], self.target[0])
        return _utils.plot_one(self.sample[0])

    @property
    def desc(self):
        return f"Rec Loss: {self.rec_loss.item():.4f}, KL Loss: {self.kl_loss.item():.4f}, Loss: {self.loss.item():.4f}"


class VectorQuantization:
    quantized_embeddings: Tensor
    quantized_mapping: Tensor
    vq_loss: Tensor

    def __init__(self, latent_states: Tensor, embeddings: Tensor):
        shape = latent_states.shape[2:]
        latent_states = latent_states.flatten(2).transpose(-1, -2)
        with torch.no_grad():
            w = torch.cdist(latent_states, embeddings, p=2)
            self.quantized_mapping = torch.zeros_like(w).scatter_(-1, w.argmin(dim=-1, keepdim=True), 1)

        quant_latent_states = torch.einsum("bnk,kc->bnc", self.quantized_mapping, embeddings)
        self.quantized_embeddings = latent_states + (quant_latent_states - latent_states).detach()
        self.vq_loss = F.mse_loss(latent_states, quant_latent_states)
        self.quantized_embeddings = self.quantized_embeddings.transpose(-1, -2).reshape(latent_states.shape[0], -1, *shape)


class VectorQuantizedVariationalAutoEncoder(nn.Module):

    in_channels: int
    base_channels: int
    latent_channels: int
    num_layers: int
    encoder: nn.Module
    latent_embeddings: nn.ParameterDict
    decoder: nn.Module

    def encode(self, input: Tensor) -> VectorQuantization:
        latent_states = self.encoder(input)
        return VectorQuantization(latent_states, self.latent_embeddings["weight"])

    def decode(self, sample: Tensor) -> Tensor:
        return self.decoder(sample)

    def forward(
        self, input: Tensor, target: Optional[Tensor] = None, kl_loss_weight: float = 0.1
    ) -> VectorQuantizedVariationalAutoEncoderOutput:
        latent_dist = self.encode(input)
        sample = self.decode(latent_dist.quantized_embeddings)
        rec_loss = F.mse_loss(sample, target)
        loss = rec_loss + kl_loss_weight * latent_dist.vq_loss
        return VectorQuantizedVariationalAutoEncoderOutput(sample, input, latent_dist, rec_loss, latent_dist.vq_loss, loss)


class VectorQuantizedVariationalAutoEncoder2d(VectorQuantizedVariationalAutoEncoder):

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 4,
        num_layers: int = 3,
        embedding_size: int = 8192,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels
        self.num_layers = num_layers
        self.latent_embeddings = nn.ParameterDict(
            {"weight": nn.Parameter(torch.empty(embedding_size, latent_channels, device=device, dtype=dtype), requires_grad=True)}
        )
        nn.init.normal_(self.latent_embeddings["weight"], mean=0, std=1 / math.sqrt(latent_channels))

        self.encoder = UnetEncoder2d(
            in_channels,
            latent_channels,
            False,
            False,
            True,
            base_channels,
            2,
            num_layers,
            device=device,
            dtype=dtype,
        )
        self.decoder = UnetDecoder2d(
            in_channels,
            latent_channels,
            False,
            False,
            True,
            base_channels,
            2,
            num_layers,
            device=device,
            dtype=dtype,
        )


class VectorQuantizedVariationalAutoEncoder3d(VectorQuantizedVariationalAutoEncoder):

    def __init__(
        self, in_channels: int = 1, base_channels: int = 128, latent_channels: int = 4, num_layers: int = 3, device=None, dtype=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels
        self.num_layers = num_layers

        self.encoder = UnetEncoder3d(in_channels, latent_channels, False, True, base_channels, 2, num_layers, device=device, dtype=dtype)
        self.decoder = UnetDecoder3d(in_channels, latent_channels, False, True, base_channels, 2, num_layers, device=device, dtype=dtype)


VQVAE = VectorQuantizedVariationalAutoEncoder
VQVAE2d = VectorQuantizedVariationalAutoEncoder2d
VQVAE3d = VectorQuantizedVariationalAutoEncoder3d
