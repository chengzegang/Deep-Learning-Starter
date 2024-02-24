from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import init_device_mesh
import torch
from torch import nn, Tensor


def setup_distributed(model: nn):
    tp_mesh = init_device_mesh("cuda", (8,))
    model = parallelize_module(model, tp_mesh, {"w1": ColwiseParallel(), "w2": RowwiseParallel()})
    return model
