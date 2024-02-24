from argparse import Namespace
import math
from deep_learning_starter.modules import UnetEncoder2d, UnetDecoder2d, UnetEncoder3d, UnetDecoder3d
import torch
from torch import nn, Tensor
from tensordict import tensorclass
import torch.nn.functional as F


class DiagonalGaussianDistribution:
    latent_states: Tensor
    mean: Tensor
    logvar: Tensor

    def __init__(self, latent_states: Tensor):
        self.latent_states = latent_states
        self.mean, self.logvar = torch.chunk(latent_states, 2, dim=1)
        logvar_dtype = self.logvar.dtype
        clamp_logvar = torch.clamp(self.logvar, -math.log(torch.finfo(logvar_dtype).eps), math.log(torch.finfo(logvar_dtype).max))
        self.logvar = self.logvar + (clamp_logvar - self.logvar).detach()

    @property
    def sample(self) -> Tensor:
        return self.mean + torch.exp(0.5 * self.logvar) * torch.randn_like(self.mean)

    @property
    def kl_loss(self) -> Tensor:
        return -0.5 * torch.mean(1 + self.logvar - self.mean.pow(2) - self.logvar.exp(), dim=-1)


class VariationalAutoEncoder(nn.Module):

    in_channels: int
    base_channels: int
    latent_channels: int
    num_layers: int
    encoder: nn.Module
    decoder: nn.Module

    def encode(self, input: Tensor) -> DiagonalGaussianDistribution:
        latent_states = self.encoder(input)
        return DiagonalGaussianDistribution(latent_states)

    def decode(self, sample: Tensor) -> Tensor:
        return self.decoder(sample)

    def forward(self, input: Tensor, kl_loss_weight: float = 1.0) -> Namespace:
        latent_dist = self.encode(input)
        latent_sample = latent_dist.sample
        sample = self.decode(latent_sample)
        rec_loss = F.mse_loss(sample, input)
        kl_loss = latent_dist.kl_loss
        loss = rec_loss + kl_loss_weight * kl_loss
        return Namespace(sample=sample, latent_dist=latent_dist, rec_loss=rec_loss, kl_loss=kl_loss, loss=loss)


class VariationalAutoEncoder2d(VariationalAutoEncoder):

    def __init__(self, in_channels: int, base_channels: int, latent_channels: int, num_layers: int, device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels
        self.num_layers = num_layers

        self.encoder = UnetEncoder2d(in_channels, latent_channels, False, True, base_channels, 2, num_layers, device=device, dtype=dtype)
        self.decoder = UnetDecoder2d(
            latent_channels, latent_channels, False, True, base_channels, 2, num_layers, device=device, dtype=dtype
        )


class VariationalAutoEncoder3d(VariationalAutoEncoder):

    def __init__(self, in_channels: int, base_channels: int, latent_channels: int, num_layers: int, device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels
        self.num_layers = num_layers

        self.encoder = UnetEncoder3d(in_channels, latent_channels, False, True, base_channels, 2, num_layers, device=device, dtype=dtype)
        self.decoder = UnetDecoder3d(
            latent_channels, latent_channels, False, True, base_channels, 2, num_layers, device=device, dtype=dtype
        )
