import torch
import torch.nn.functional as F
from torch import nn
from typing import Callable
import matplotlib.pyplot as plt


class SRLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = self.__class__.__name__

    @property
    def best_min(self) -> bool:
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    def forward(self, sr, hr, hr_mask=None):
        assert type(sr) == type(hr), f"Super-resolved image ({type(sr)}) is not the same type" \
                                     f" as target HR image ({type(hr)})!"
        if isinstance(sr, dict):
            if hr_mask is None:
                hr_mask = {k: None for k in sr.keys()}
            results = {k: self._forward(sr[k], hr[k], hr_mask[k]) for k in sr.keys()}
            return torch.mean(torch.stack(list(results.values())))
        return self._forward(sr, hr, hr_mask)

    def _forward(self, sr, hr, hr_mask=None):
        raise NotImplementedError

    @staticmethod
    def mask_pixels(sr, hr, hr_mask=None):
        if hr_mask is None:
            hr_mask = torch.ones_like(hr)
        sr = sr*hr_mask
        hr = hr*hr_mask
        total_unmasked = torch.sum(hr_mask, dim=(-3, -2, -1))
        return sr, hr, total_unmasked


class MSE(SRLoss):
    @property
    def best_min(self) -> bool:
        return True

    def __init__(self):
        super().__init__()

    def _forward(self, sr: torch.Tensor, hr: torch.Tensor, hr_mask: torch.Tensor = None):
        """
        Masked version of MSE (L2) loss.
        """
        sr, hr, total_unmasked = self.mask_pixels(sr, hr, hr_mask)
        result = torch.sum(torch.square(hr - sr), dim=(-3, -2, -1)) / total_unmasked
        return result


class L1(SRLoss):
    @property
    def best_min(self) -> bool:
        return True

    def __init__(self):
        super().__init__()

    def _forward(self, sr: torch.Tensor, hr: torch.Tensor, hr_mask: torch.Tensor = None):
        """
        Masked version of MAE (L1) loss.
        """
        sr, hr, total_unmasked = self.mask_pixels(sr, hr, hr_mask)
        l1_loss = torch.sum(torch.abs(hr-sr), dim=(-3, -2, -1)) / total_unmasked
        return l1_loss


class cMSE(SRLoss):
    @property
    def best_min(self) -> bool:
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = MSE()

    def _forward(self, sr: torch.Tensor, hr: torch.Tensor, hr_mask: torch.Tensor = None, **kwargs):
        if hr_mask is None:
            return self.unmasked_forward(sr, hr)
        return self.masked_forward(sr, hr, hr_mask)

    def unmasked_forward(self, sr: torch.Tensor, hr: torch.Tensor):
        hr_size = hr.shape[-1]
        border = 3
        max_pixel_shifts = 2 * border
        size_cropped_image = hr_size - max_pixel_shifts
        cropped_predictions = sr[:, :, border:hr_size - border, border:hr_size - border]
        x = []

        for i in range(max_pixel_shifts + 1):
            for j in range(max_pixel_shifts + 1):
                cropped_labels = hr[:, :, i:i + size_cropped_image,
                                 j:j + size_cropped_image]
                total_pixels = cropped_labels.numel()
                b = (1.0 / total_pixels) * torch.sum(torch.sub(cropped_labels, cropped_predictions), dim=[1, 2, 3])
                b = b.view(-1, 1, 1, 1)
                corrected_cropped_predictions = cropped_predictions + b
                corrected_mse = (1.0 / total_pixels) * torch.sum(
                    torch.square(
                        torch.sub(cropped_labels, corrected_cropped_predictions)
                    ), dim=[1, 2, 3])
                x.append(corrected_mse)
        x = torch.stack(x)
        min_cmse = torch.min(x, 0)[0]
        return torch.mean(min_cmse)

    def masked_forward(self, sr: torch.Tensor, hr: torch.Tensor, hr_mask: torch.Tensor):
        hr_size = hr.shape[-1]
        border = 3
        max_pixel_shifts = 2 * border
        crop_size = hr_size - max_pixel_shifts
        sr = sr * hr_mask
        cropped_predictions = (sr[:, :, border:hr_size - border, border:hr_size - border]).type(torch.float32)
        hr = (hr * hr_mask).type(torch.float32)
        x = []

        for i in range(max_pixel_shifts + 1):
            for j in range(max_pixel_shifts + 1):
                cropped_labels = hr[:, :, i:i + crop_size,
                                 j:j + crop_size]
                cropped_hr_mask = hr_mask[:, :, i:i + crop_size,
                                  j:j + crop_size].type(torch.float32)
                total_pixels_masked = torch.sum(cropped_hr_mask, dim=[1, 2, 3])
                b = (1.0 / total_pixels_masked) * torch.sum(torch.sub(cropped_labels,
                                                                      cropped_predictions), dim=[1, 2, 3])
                b = b.view(-1, 1, 1, 1)
                corrected_cropped_predictions = (cropped_predictions + b) * cropped_hr_mask
                corrected_mse = (1.0 / total_pixels_masked) * torch.sum(
                    torch.square(
                        torch.sub(cropped_labels, corrected_cropped_predictions)
                    ), dim=[1, 2, 3])
                x.append(corrected_mse)
        x = torch.stack(x)
        min_cmse = torch.min(x, 0)[0]
        return torch.mean(min_cmse)