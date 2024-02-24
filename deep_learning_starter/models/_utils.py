import matplotlib.pyplot as plt
import torchvision.transforms.v2.functional as TF
import torch.nn.functional as F
from torch import Tensor


def plot_pair(pred: Tensor, target: Tensor, fig_size: int = 4):
    fig, ax = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size))
    ax[0].imshow(TF.to_pil_image(pred.float().detach().cpu().clamp(0, 1)))
    ax[1].imshow(TF.to_pil_image(target.float().detach().cpu().clamp(0, 1)))
    ax[0].set_title("Prediction")
    ax[1].set_title("Target")
    ax[0].axis("off")
    ax[1].axis("off")
    fig.tight_layout()
    return fig


def plot_one(pred: Tensor, fig_size: int = 4):
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
    ax.imshow(TF.to_pil_image(pred.float().detach().cpu().clamp(0, 1)))
    ax.axis("off")
    fig.tight_layout()
    return fig
