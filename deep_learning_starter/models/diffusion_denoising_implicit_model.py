import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F
from diffusers import DDIMScheduler
from dataclasses import dataclass


@dataclass
class LatentDiffusionText2ImageOutput:
    loss: Tensor
    original_sample: Tensor
    pred_original_sample: Tensor


class LatentDiffusionText2Image(nn.Module):

    def __init__(self, autoencoder: nn.Module, diffusion_model: nn.Module, num_training_steps: int = 1000, **kwargs):
        super().__init__()
        self.autoencoder = autoencoder
        self.diffusion_model = diffusion_model
        self.scheduler = DDIMScheduler(num_training_steps=num_training_steps, **kwargs)
        self.num_training_steps = num_training_steps

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            latent_states = self.autoencoder.encode(input)
            t = torch.randint(0, self.scheduler.config.num_training_steps, (input.size(0),), device=input.device)
            z = latent_states.quantized_mappings
            eps_t = torch.randn_like(z)
            z_t = self.scheduler.add_noise(z, eps_t, t)
            z_t = self.scheduler.scale_model_input(z_t, t)
        eps_t_hat = self.diffusion_model(z_t, t)
        loss = F.mse_loss(eps_t_hat, eps_t)
        with torch.no_grad():
            self.scheduler.set_timesteps(self.num_training_steps)
            z_hat = self.scheduler.step(eps_t_hat, t, z_t).pred_original_sample
            pred_original_sample = self.autoencoder.decode(z_hat)
        return LatentDiffusionText2ImageOutput(loss, input, pred_original_sample)
