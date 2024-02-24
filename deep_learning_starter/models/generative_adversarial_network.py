__all__ = ["GenerativeAdversialNetwork", "GenerativeAdversialNetworkOutput", "GAN"]
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from deep_learning_starter.modules import UnetDecoder2d, UnetEncoder2d
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from . import _utils
from deep_learning_starter.modules.rmsnorm import SpatialRMSNorm


@dataclass
class GenerativeAdversialNetworkOutput:
    sample: Tensor
    g_loss: Tensor
    d_loss: Tensor
    loss: Tensor

    @property
    def plot(self):
        return _utils.plot_one(self.sample[0])

    @property
    def desc(self):
        return f"G Loss: {self.g_loss.item():.4f}, D Loss: {self.d_loss.item():.4f}, Loss: {self.loss.item():.4f}"


class ReplayBuffer:

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data = []

    def sample(self, sample_size: int = 1):
        index = np.random.choice(len(self.data), sample_size, replace=False)
        return [self.data[i] for i in index]

    def push(self, samples: Tuple[Tensor, ...]):
        self.data.extend(samples)
        if len(self.data) > self.max_size:
            sample_index = np.random.choice(len(self.data), len(self.data) - self.max_size, replace=False)
            self.data = [sample for i, sample in enumerate(self.data) if i not in sample_index]


class GenerativeAdversialNetwork(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        latent_channels: int = 128,
        num_layers: int = 4,
        eps: float = 1e-8,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels
        self.num_layers = num_layers

        self.generator = UnetDecoder2d(
            in_channels,
            latent_channels,
            False,
            False,
            True,
            base_channels,
            2,
            num_layers,
            activation=nn.ReLU(True),
            normalization=nn.BatchNorm2d,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self.discriminator = UnetEncoder2d(
            in_channels,
            1,
            False,
            False,
            True,
            base_channels,
            2,
            num_layers,
            activation=nn.LeakyReLU(0.2),
            normalization=nn.BatchNorm2d,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        self.replay_buffer = ReplayBuffer(1000)

    def sample(self, sample_shape: Tuple[int, ...], device=None, dtype=None) -> Tensor:
        latent_shape = (
            sample_shape[:-3] + (self.latent_channels,) + (sample_shape[-2] // 2**self.num_layers, sample_shape[-1] // 2**self.num_layers)
        )
        seed = torch.randn(latent_shape, device=device, dtype=dtype).clamp(-1, 1)
        sample = self.generator(seed).clamp(0, 1)
        return sample

    def critic(self, sample: Tensor) -> Tensor:
        logits = self.discriminator(sample)
        return logits

    def forward(self, input: Tensor, target: Optional[Tensor] = None) -> GenerativeAdversialNetworkOutput:

        self.generator.requires_grad_(True)
        self.discriminator.requires_grad_(False)
        sample = self.sample(input.shape, device=input.device, dtype=input.dtype)
        self.replay_buffer.push(sample.detach().cpu().unbind())
        fake_logits = self.critic(sample)
        g_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

        self.generator.requires_grad_(False)
        self.discriminator.requires_grad_(True)
        with torch.no_grad():
            sample = self.sample(input.shape, device=input.device, dtype=input.dtype)
            replay_sample = torch.stack(self.replay_buffer.sample(input.shape[0]), dim=0).type_as(sample)
            sample = torch.cat([sample, replay_sample], dim=0)
        fake_logits = self.critic(sample.detach())
        real_logits = self.critic(input)
        d_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )

        return GenerativeAdversialNetworkOutput(
            sample=sample, g_loss=g_loss, d_loss=d_loss, loss=g_loss + d_loss / (g_loss / d_loss).detach()
        )


GAN = GenerativeAdversialNetwork
