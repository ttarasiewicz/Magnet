from .test_model import SplineModel, TestModel
from .FSRCNN import FSRCNN, GraphFSRCNN
from .RAMS import RAMS
from .MagNet import MagNet, MagNetv2
from .model import Model, GraphModel
from .HighResNet import *

import torch.nn.functional as F


class Bicubic(Model):
    def __init__(self, scale=3, device='cuda'):
        super().__init__()
        self.scale = scale
        self.device = device

    def forward(self, data):
        x = data.x
        img = torch.zeros((1, 1, x.shape[-2] * 3, x.shape[-1] * 3)).to(self.device)
        for i in range(x.shape[1]):
            example = x[0:1, i:i + 1, :, :]
            img += F.interpolate(example, scale_factor=self.scale, mode='bicubic', align_corners=False)
        img = img / x.shape[1]
        return img

    def load_state_dict(self, state_dict, strict=False):
        return self


class NearestNeighbors(Model):
    def __init__(self, scale=3, device='cuda'):
        super().__init__()
        self.scale = scale
        self.device = device

    def forward(self, data):
        x = data.x
        img = torch.zeros((1, 1, x.shape[-2] * 3, x.shape[-1] * 3)).to(self.device)
        # for i in range(x.shape[1]):
        for i in range(1):
            example = x[0:1, i:i + 1, :, :]
            img += F.interpolate(example, scale_factor=self.scale, mode='nearest')
        img = img / x.shape[1]
        return img

    def load_state_dict(self, state_dict, strict=False):
        return self
