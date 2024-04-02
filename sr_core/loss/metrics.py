from sr_core.loss import SRLoss, MSE
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable
from enum import Enum
# from image_similarity_measures.quality_metrics import psnr, rmse, ssim, uiq


# class MetricMode(Enum):
#     MAX = 0
#     MIN = 1
#
#
# _mode_per_metric = {psnr: np.max,
#                     ssim: np.max,
#                     rmse: np.min,
#                     uiq: np.max
#                     }
#
#
# class cBase(nn.Module):
#     def __init__(self, loss: Callable):
#         super().__init__()
#         self.loss = loss
#         self.reduce = _mode_per_metric.get(loss)
#
#     def forward(self, sr: torch.Tensor, hr: torch.Tensor):
#         assert sr.shape[0] == 1 and sr.ndim == 4, f"Wrong shape of input image ({sr.shape})"
#         border = 3
#         max_pixel_shifts = 2 * border
#         size_cropped_image = hr.shape[-2] - max_pixel_shifts, hr.shape[-1] - max_pixel_shifts
#         cropped_predictions = sr[:, :, border:hr.shape[-2] - border, border:hr.shape[-1] - border]
#         zscore = (cropped_predictions - cropped_predictions.mean()) / (cropped_predictions.std() + 1e-15)
#         x = []
#
#         for i in range(max_pixel_shifts + 1):
#             for j in range(max_pixel_shifts + 1):
#                 cropped_labels = hr[:, :, i:i + size_cropped_image[0],
#                                 j:j + size_cropped_image[1]]
#                 cropped_predictions = zscore * cropped_labels.std() + cropped_labels.mean()
#                 total_pixels = cropped_labels.numel()
#                 b = (1.0 / total_pixels) * torch.sum(torch.sub(cropped_labels, cropped_predictions), dim=[1, 2, 3])
#                 b = b.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#                 corrected_cropped_predictions = (cropped_predictions + b)
#                 corrected_cropped_predictions = corrected_cropped_predictions.clamp(0., 1.)
#                 cropped_labels = cropped_labels.detach().cpu().numpy()[0, :, :, :]
#                 cropped_labels = np.moveaxis(cropped_labels, 0, -1)
#                 corrected_cropped_predictions = corrected_cropped_predictions.detach().cpu().numpy()[0, :, :, :]
#                 corrected_cropped_predictions = np.moveaxis(corrected_cropped_predictions, 0, -1)
#                 loss = self.loss(cropped_labels,
#                                  corrected_cropped_predictions,
#                                  max_p=1.0)
#                 x.append(loss)
#         x = np.stack(x)
#         best_score = self.reduce(x, 0)
#         return np.mean(best_score)
#
#     @property
#     def name(self):
#         return self.loss.__name__


class PSNR(MSE):
    @property
    def best_min(self) -> bool:
        return False

    def __init__(self):
        super().__init__()

    def _forward(self, sr: torch.Tensor, hr: torch.Tensor, hr_mask: torch.Tensor = None):
        return -10 * torch.log10(super()._forward(sr, hr, hr_mask))


class SAM(SRLoss):
    @property
    def best_min(self) -> bool:
        return True

    def _forward(self, sr: torch.Tensor, hr: torch.Tensor, hr_mask: torch.Tensor = None):
        sr, hr, total_unmasked = self.mask_pixels(sr, hr, hr_mask)
        sr_norm = torch.sqrt(sr * sr)
        hr_norm = torch.sqrt(hr * hr)
        result = torch.sum(torch.acos(sr * hr / (sr_norm * hr_norm + 1e-10)), dim=(-3, -2, -1)) / total_unmasked
        return result


class SSIM(SRLoss):
    @property
    def best_min(self) -> bool:
        return False

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    @staticmethod
    def gaussian(window_size, sigma):
        from math import exp
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @staticmethod
    def create_window(window_size, channel):
        from torch.autograd import Variable
        _1D_window = SSIM.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map
        # if size_average:
        #     return ssim_map.mean()
        # else:
        #     return ssim_map.mean(1).mean(1).mean(1)

    def _forward(self, sr: torch.Tensor, hr: torch.Tensor, hr_mask: torch.Tensor = None):
        input_shape = sr.shape
        sr, hr, _ = self.mask_pixels(sr, hr, hr_mask)
        sr = sr.view(-1, *input_shape[-3:])
        hr = hr.view(-1, *input_shape[-3:])
        (_, channel, _, _) = sr.size()

        if channel == self.channel and self.window.data.type() == sr.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if sr.is_cuda:
                window = window.cuda(sr.get_device())
            window = window.type_as(sr)

            self.window = window
            self.channel = channel
        result = self._ssim(sr, hr, window, self.window_size, channel, self.size_average).mean((-3, -2, -1))
        result = result.view(input_shape[:-3])
        return result
