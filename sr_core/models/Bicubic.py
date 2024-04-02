import torch.nn.functional as F
from sr_core.models import Model


class Bicubic(Model):
    def __init__(self, scale=3):
        super().__init__()
        self.scale = scale

    def forward(self, entry):
        x = entry.lr_images
        if entry.hr_image is not None:
            scale = round(entry.hr_image.shape[-1] / x.shape[-1])
        else:
            scale = self.scale
        img = x.new_zeros((1, 1, x.shape[-2] * scale, x.shape[-1] * scale))
        for i in range(x.shape[1]):
            example = x[0:1, i:i + 1, :, :]
            img += F.interpolate(example, scale_factor=scale, mode='bicubic', align_corners=False)
        img = img / x.shape[1]
        return img
