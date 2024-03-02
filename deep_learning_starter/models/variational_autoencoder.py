__all__ = [
    "VariationalAutoEncoder",
    "VariationalAutoEncoder2d",
    "VariationalAutoEncoder3d",
    "VariationalAutoEncoderOutput",
    "DiagonalGaussianDistribution",
    "VAE2d",
    "VAE3d",
]
import math
from typing import Optional
from deep_learning_starter.modules import UnetEncoder2d, UnetDecoder2d, UnetEncoder3d, UnetDecoder3d
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from deep_learning_starter.modules.rmsnorm import SpatialRMSNorm
from . import _utils
from kornia.color import rgb_to_lab
from dataclasses import dataclass
import torchvision.transforms.v2.functional as TF


@dataclass
class VariationalAutoEncoderOutput:
    sample: Tensor
    target: Optional[Tensor]
    latent_dist: "DiagonalGaussianDistribution"
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


@dataclass(init=False)
class DiagonalGaussianDistribution:
    latent_states: Tensor
    mean: Tensor
    logvar: Tensor

    def __init__(self, latent_states: Tensor):
        self.latent_states = latent_states
        self.mean, self.logvar = torch.chunk(latent_states, 2, dim=1)
        clamp_logvar = torch.clamp(self.logvar, -30, 20)
        self.logvar = self.logvar + (clamp_logvar - self.logvar).detach()

    @property
    @torch.autocast("cuda", torch.float32)
    def sample(self) -> Tensor:
        return self.mean + torch.exp(0.5 * self.logvar) * torch.randn_like(self.mean)

    @property
    @torch.autocast("cuda", torch.float32)
    def kl_loss(self) -> Tensor:
        return -0.5 * torch.mean(1 + self.logvar - self.mean.pow(2) - self.logvar.exp())


class VariationalAutoEncoder(nn.Module):

    in_channels: int
    base_channels: int
    latent_channels: int
    num_layers: int
    encoder: nn.Module
    decoder: nn.Module
    _feature_extractor: nn.Module

    def encode(self, input: Tensor) -> DiagonalGaussianDistribution:
        latent_states = self.encoder(input)
        return DiagonalGaussianDistribution(latent_states)

    def decode(self, sample: Tensor) -> Tensor:
        return self.decoder(sample)

    def forward(self, input: Tensor, target: Optional[Tensor] = None, kl_loss_weight: float = 0.1) -> VariationalAutoEncoderOutput:
        latent_dist = self.encode(input)
        latent_sample = latent_dist.sample
        sample = self.decode(latent_sample)
        rec_loss = F.l1_loss(sample, input)
        nsample = TF.normalize(sample, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ninput = TF.normalize(input, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        sample_feats = self._feature_extractor(nsample, output_hidden_states=True).hidden_states
        input_feats = self._feature_extractor(ninput, output_hidden_states=True).hidden_states
        for sf, inf in zip(sample_feats, input_feats):
            rec_loss += F.mse_loss(sf, inf)
        kl_loss = latent_dist.kl_loss
        loss = rec_loss + kl_loss_weight * kl_loss
        return VariationalAutoEncoderOutput(sample, input, latent_dist, rec_loss, kl_loss, loss)


class VariationalAutoEncoder2d(VariationalAutoEncoder):

    def __init__(
        self, in_channels: int = 3, base_channels: int = 64, latent_channels: int = 12, num_layers: int = 3, device=None, dtype=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels
        self.num_layers = num_layers

        self.encoder = UnetEncoder2d(
            in_channels,
            latent_channels * 2,
            False,
            False,
            True,
            base_channels,
            2,
            num_layers,
            bias=True,
            device=device,
            dtype=dtype,
            max_seq_length=16384,
            normalization=SpatialRMSNorm,
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
            bias=True,
            device=device,
            dtype=dtype,
            max_seq_length=16384,
            normalization=SpatialRMSNorm,
        )

        from transformers import ResNetBackbone

        self._feature_extractor = ResNetBackbone.from_pretrained("microsoft/resnet-50", torch_dtype=dtype, device_map=device)
        self._feature_extractor.requires_grad_(False)


class VariationalAutoEncoder3d(VariationalAutoEncoder):

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


VAE2d = VariationalAutoEncoder2d
VAE3d = VariationalAutoEncoder3d
