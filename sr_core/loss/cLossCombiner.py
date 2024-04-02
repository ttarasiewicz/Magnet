from . import SRLoss
from typing import Union, List
from itertools import product
import torch
from torch.nn import Module


class LossCombinerBase(SRLoss):
    @property
    def best_min(self):
        return None

    def __init__(self, losses: Union[SRLoss, List[SRLoss]], training_mode: bool = True):
        super().__init__()
        self.train(training_mode)
        if not isinstance(losses, list):
            losses = [losses]
        self.losses = losses
        self._rename_losses()

    def __iter__(self):
        return iter(self.losses)

    def _forward(self, sr, hr, hr_mask=None):
        raise NotImplementedError

    def _rename_losses(self):
        raise NotImplementedError


class cLossCombiner(LossCombinerBase):
    def _rename_losses(self):
        for loss in self.losses:
            loss.name = f"c{loss.name}"

    def __init__(self, losses: Union[SRLoss, List[SRLoss]], training_mode: bool = True):
        super().__init__(losses, training_mode)
        self.loss_choice_functions = {loss: torch.min if loss.best_min else torch.max for loss in self.losses}

    def _forward(self, sr: torch.Tensor, hr: torch.Tensor, hr_mask: torch.Tensor = None):
        height, width = hr.shape[-2:]
        if hr_mask is None:
            hr_mask = torch.ones_like(hr)
        border = 3
        max_pixel_shifts = 2 * border
        cropped_height, cropped_width = height - max_pixel_shifts, width - max_pixel_shifts
        sr_patch = sr[:, :, border:height - border, border:width - border]
        hr_patches = []
        mask_patches = []
        for i, j in product(range(max_pixel_shifts + 1), range(max_pixel_shifts + 1)):
            hr_patches.append(hr[:, :, i:i + cropped_height,
                              j:j + cropped_width])
            mask_patches.append(hr_mask[:, :, i:i + cropped_height,
                                j:j + cropped_width])

        hr_patches = torch.stack(hr_patches, dim=1)
        mask_patches = torch.stack(mask_patches, dim=1)
        sr_patch = torch.repeat_interleave(sr_patch.unsqueeze(1), hr_patches.shape[1], dim=1)
        total_unmasked = torch.sum(mask_patches, dim=[2, 3, 4]).view(-1, hr_patches.shape[1], 1, 1, 1)
        b = torch.pow(total_unmasked, -1) * torch.sum(hr_patches-sr_patch, dim=[2, 3, 4]).view(total_unmasked.shape)
        sr_patch = sr_patch + b
        results = {}
        for loss, choice_function in self.loss_choice_functions.items():
            loss_values = loss(sr_patch, hr_patches, mask_patches)
            results[loss.name] = torch.mean(choice_function(loss_values, dim=1)[0])
        return results
