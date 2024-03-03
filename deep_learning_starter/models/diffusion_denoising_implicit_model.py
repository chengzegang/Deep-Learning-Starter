from typing import List
import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F
from diffusers import DDIMScheduler
from dataclasses import dataclass

from deep_learning_starter.modules.transformer import Transformer
import open_clip
from . import _utils


@dataclass
class LatentDiffusionText2ImageOutput:
    loss: Tensor
    timestep: Tensor
    original_sample: Tensor
    pred_original_sample: Tensor

    @property
    def plot(self):
        if self.original_sample is not None:
            return _utils.plot_pair(self.pred_original_sample[0], self.original_sample[0], title=f"t={self.timestep[0].item()}")
        return _utils.plot_one(self.pred_original_sample[0])

    @property
    def desc(self):
        return f"Loss: {self.loss.item():.4f}"


class LatentDiffusionTransformer(nn.Module):

    def __init__(
        self,
        hidden_size: int = 512,
        head_size: int = 64,
        num_heads: int = 8,
        num_layers: int = 16,
        eps: float = 1e-8,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.eps = eps
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K", precision="fp16", device=device
        )
        self.clip.eval()
        self.clip.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        self.timestep_embedding = nn.Embedding(1000, hidden_size, device=device, dtype=dtype, padding_idx=0)
        self.in_conv = nn.Conv2d(12, hidden_size, kernel_size=3, stride=1, padding=1, bias=False, device=device, dtype=dtype)
        self.decoder = Transformer(
            hidden_size=hidden_size, head_size=head_size, num_heads=num_heads, num_layers=num_layers, eps=eps, device=device, dtype=dtype
        )
        self.out_conv = nn.Conv2d(hidden_size, 12, kernel_size=3, stride=1, padding=1, bias=False, device=device, dtype=dtype)
        self.latent_scale = 0.125

    def forward(self, image: Tensor, text: List[str], timestep: Tensor) -> Tensor:
        with torch.no_grad():
            text = self.tokenizer(text).to(image.device)
            text = self.clip.encode_text(text)
        timestep = self.timestep_embedding(timestep.view(-1, 1))
        image = self.in_conv(image)
        image_embeds = image.flatten(2).transpose(-1, -2)
        token_embeds = torch.cat([image_embeds, text.unsqueeze(1), timestep], dim=1)
        output_embeds = self.decoder(token_embeds)
        output_embeds = output_embeds[:, : image_embeds.size(1)]
        output_embeds = output_embeds.transpose(-1, -2).view_as(image)
        output_embeds = self.out_conv(output_embeds)
        return output_embeds


class LatentDiffusionText2Image(nn.Module):

    def __init__(self, autoencoder: nn.Module, num_train_timesteps: int = 1000, **kwargs):
        super().__init__()
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        self.autoencoder.requires_grad_(False)
        self.diffusion_model = LatentDiffusionTransformer(**kwargs)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
        self.num_train_timesteps = num_train_timesteps
        self.latent_scale = 0.125

    def forward(self, image: Tensor, text: List[str]) -> Tensor:
        with torch.no_grad():
            latent_states = self.autoencoder.encode(image)
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (1,), device=image.device)
            z = latent_states.sample * self.latent_scale
            eps_t = torch.randn_like(z)
            z_t = self.scheduler.add_noise(z, eps_t, t)
            z_t = self.scheduler.scale_model_input(z_t, t)
        eps_t_hat = self.diffusion_model(z_t, text, t.expand(z_t.size(0), -1))
        loss = F.mse_loss(eps_t_hat, eps_t)
        with torch.no_grad():
            self.scheduler.set_timesteps(self.num_train_timesteps)
            z_hat = self.scheduler.step(eps_t_hat, t, z_t).pred_original_sample / self.latent_scale
            pred_original_sample = self.autoencoder.decode(z_hat)
        return LatentDiffusionText2ImageOutput(loss, t, image, pred_original_sample)

    def generate(self, text: str) -> Tensor:
        with torch.no_grad():
            z = torch.randn(1, self.autoencoder.latent_channels, device=self.diffusion_model.device, dtype=self.diffusion_model.dtype)
            z = z * self.latent_scale
            z = self.scheduler.scale_model_input(z, 0)
            z = self.scheduler.step(z, 0).pred_original_sample / self.latent_scale
            pred_original_sample = self.autoencoder.decode(z)
        return pred_original_sample
