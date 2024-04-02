import torch
from torch import nn
from torch.nn import functional as F
from sr_core.models.model import Model


class RFAB(nn.Module):
    def __init__(self, filters, kernel_size, r):
        super().__init__()
        self.pre_attention = nn.Sequential(
            nn.utils.weight_norm(nn.Conv3d(in_channels=filters, out_channels=filters,
                                           kernel_size=kernel_size, padding=kernel_size // 2)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv3d(in_channels=filters, out_channels=filters,
                                           kernel_size=kernel_size, padding=kernel_size // 2))
        )

        self.feature_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.utils.weight_norm(nn.Conv3d(in_channels=filters, out_channels=filters // r,
                                           kernel_size=kernel_size, padding=kernel_size // 2)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv3d(in_channels=filters // r, out_channels=filters,
                                           kernel_size=kernel_size, padding=kernel_size // 2)),
        )

    def forward(self, x):
        x_rfab_res = x
        x = self.pre_attention(x)
        x_fa_res = x
        x = self.feature_attention(x)
        x = x * x_fa_res
        return x + x_rfab_res


class TemporalReductionBlock(nn.Module):
    def __init__(self, filters, kernel_size, r):
        super().__init__()
        self.rfab = RFAB(filters, kernel_size, r)
        self.conv = nn.Sequential(
            nn.utils.weight_norm(nn.Conv3d(in_channels=filters, out_channels=filters,
                                           kernel_size=kernel_size)),
            nn.ReLU()
        )

    def forward(self, x):
        x = F.pad(x, [1, 1, 1, 1, 0, 0], mode='replicate')
        x = self.rfab(x)
        x = self.conv(x)
        return x


class RTAB(nn.Module):
    def __init__(self, filters, kernel_size, r):
        super().__init__()
        self.attention = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(in_channels=filters, out_channels=filters,
                                           kernel_size=kernel_size, padding=kernel_size // 2)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv2d(in_channels=filters, out_channels=filters,
                                           kernel_size=kernel_size, padding=kernel_size // 2))
        )
        self.temporal = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.utils.weight_norm(nn.Conv2d(in_channels=filters, out_channels=filters // r,
                                           kernel_size=kernel_size, padding=kernel_size // 2)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv2d(in_channels=filters // r, out_channels=filters,
                                           kernel_size=kernel_size, padding=kernel_size // 2)),
        )

    def forward(self, x):
        x_res = x
        x = self.attention(x)
        x_temp_res = x
        x = self.temporal(x)
        x = x_temp_res * x
        x = x + x_res
        return x


class RAMS(Model):
    def __init__(self, scale=3, filters=32, kernel_size=3, channels=9, r=8, N=12):
        super().__init__()
        self.scale = scale
        self.channels = channels
        self.conv1 = nn.utils.weight_norm(nn.Conv3d(in_channels=1, out_channels=filters,
                                                    kernel_size=kernel_size, padding=1))
        self.rfabs = [RFAB(filters, kernel_size, r) for i in range(N)]
        self.rfabs = nn.Sequential(*self.rfabs)
        self.conv2 = nn.utils.weight_norm(nn.Conv3d(in_channels=filters, out_channels=filters,
                                                    kernel_size=kernel_size, padding=1))
        self.trbs = [TemporalReductionBlock(filters, kernel_size, r) for _ in range(channels // 3)]
        self.trbs = nn.Sequential(*self.trbs)
        self.conv3 = nn.utils.weight_norm(nn.Conv3d(in_channels=filters, out_channels=scale ** 2,
                                                    kernel_size=kernel_size, padding=0))
        self.rtab = RTAB(channels, kernel_size, r)
        self.global_conv = nn.Conv2d(channels, scale ** 2, kernel_size=kernel_size)

    def forward(self, x):
        x = x.lrs
        x_global_res = x
        x = torch.unsqueeze(x, 1)
        x = F.pad(x, [1, 1, 1, 1, 0, 0], mode='replicate')

        # Low level features extraction
        x = self.conv1(x)

        # LSC
        x_res = x
        x = self.rfabs(x)
        x = self.conv2(x)
        x = x + x_res

        # Temporal Reduction, out: CxHxW
        x = self.trbs(x)

        # Upscaling
        x = self.conv3(x)
        x = x[..., 0, :, :]
        x = F.pixel_shuffle(x, self.scale)

        # Global path
        x_global_res = F.pad(x_global_res, [1, 1, 1, 1], mode='reflect')
        x_global_res = self.rtab(x_global_res)
        x_global_res = self.global_conv(x_global_res)
        x_global_res = F.pixel_shuffle(x_global_res, self.scale)
        x = x + x_global_res
        x = torch.nn.Sigmoid()(x)
        return x
