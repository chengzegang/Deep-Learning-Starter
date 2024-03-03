from typing import Tuple
from torch.utils.data import MapDataPipe
import os
import numpy as np
import torch.nn.functional as F
from tensordict import tensorclass, MemoryMappedTensor, NonTensorData
import torchvision.transforms.v2.functional as TF
from torchvision import io
import torch
from torch import Tensor


class ImageFolder(MapDataPipe):

    def __init__(self, data_dir: str, exts: Tuple[int, int] = (".jpg", ".png"), image_size: int = 512, transform=None):
        self.data_dir = data_dir
        self.exts = exts
        self.image_size = image_size
        self.transform = transform
        self.paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(exts):
                    self.paths.append(os.path.join(root, file))
        self.paths = np.asarray(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> Tensor:
        image = io.read_image(self.paths[index], io.image.ImageReadMode.RGB)
        image = TF.resize(image, self.image_size, antialias=True)
        image = TF.to_dtype_image(image, dtype=torch.float32, scale=True)
        if self.transform is not None:
            image = self.transform(image)
        image = MemoryMappedTensor.from_tensor(image)
        return {"image": image}
