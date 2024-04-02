import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from typing import List, Union


class GridPlot:
    def __init__(self, nrow: int = 8):
        super().__init__()
        self.nrow = nrow

    def show(self, grid):
        if not isinstance(grid, list):
            grid = [grid]
        fix, axs = plt.subplots(ncols=len(grid), squeeze=False)
        for i, img in enumerate(grid):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()

    def __call__(self, images: Union[List[torch.Tensor], torch.Tensor]):
        grid = make_grid(images, nrow=self.nrow)
        self.show(grid)
