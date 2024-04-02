import torch
from torch import nn
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, channel_size: int = 64, kernel_size: int = 3):
        """
        Args:
            channel_size : int, number of hidden channels
            kernel_size : int, shape of a 2D kernel
        """

        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )

    def forward(self, x):
        """
        Args:
            x : tensor (B, C, W, H), hidden state
        Returns:
            x + residual: tensor (B, C, W, H), new hidden state
        """

        residual = self.block(x)
        return x + residual


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 2, num_layers: int = 2, kernel_size: int = 3, channel_size: int = 64):
        """
        Args:
            config : dict, configuration file
        """

        super(Encoder, self).__init__()

        in_channels = in_channels
        num_layers = num_layers
        kernel_size = kernel_size
        channel_size = channel_size
        padding = kernel_size // 2

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU())

        res_layers = [ResidualBlock(channel_size, kernel_size) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_layers)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        """
        Encodes an input tensor x.
        Args:
            x : tensor (B, C_in, W, H), input images
        Returns:
            out: tensor (B, C, W, H), hidden states
        """
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x


class RecursiveNet(nn.Module):

    def __init__(self, in_channels: int = 64, num_layers: int = 2, kernel_size: int = 3, alpha_residual: bool = True):
        """
        Args:
            config : dict, configuration file
        """

        super(RecursiveNet, self).__init__()

        self.input_channels = in_channels
        self.num_layers = num_layers
        self.alpha_residual = alpha_residual
        kernel_size = kernel_size
        padding = kernel_size // 2

        self.fuse = nn.Sequential(
            ResidualBlock(2 * self.input_channels, kernel_size),
            nn.Conv2d(in_channels=2 * self.input_channels, out_channels=self.input_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.PReLU())

    def forward(self, x, alphas=None):
        """
        Fuses hidden states recursively.
        Args:
            x : tensor (B, L, C, W, H), hidden states
            alphas : tensor (B, L, 1, 1, 1), boolean indicator (0 if padded low-res view, 1 otherwise)
        Returns:
            out: tensor (B, C, W, H), fused hidden state
        """

        batch_size, nviews, channels, width, heigth = x.shape
        parity = nviews % 2
        half_len = nviews // 2
        while half_len > 0:
            alice = x[:, :half_len]  # first half hidden states (B, L/2, C, W, H)
            bob = x[:, half_len:nviews - parity]  # second half hidden states (B, L/2, C, W, H)
            bob = torch.flip(bob, [1])

            alice_and_bob = torch.cat([alice, bob], 2)  # concat hidden states accross channels (B, L/2, 2*C, W, H)
            alice_and_bob = alice_and_bob.view(-1, 2 * channels, width, heigth)
            x = self.fuse(alice_and_bob)
            x = x.view(batch_size, half_len, channels, width, heigth)  # new hidden states (B, L/2, C, W, H)

            if self.alpha_residual:  # skip connect padded views (alphas_bob = 0)
                alphas_alice = alphas[:, :half_len]
                alphas_bob = alphas[:, half_len:nviews - parity]
                alphas_bob = torch.flip(alphas_bob, [1])
                x = alice + alphas_bob * x
                alphas = alphas_alice

            nviews = half_len
            parity = nviews % 2
            half_len = nviews // 2

        return torch.mean(x, 1)


class Decoder(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64, kernel_size: int = 3, stride: int = 3,
                 final_in_channels: int = 64, final_out_channels: int = 1, final_kernel_size: int = 1):
        """
        Args:
            config : dict, configuration file
        """

        super(Decoder, self).__init__()

        self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride),
                                    nn.PReLU())

        self.final = nn.Conv2d(in_channels=final_in_channels,
                               out_channels=final_out_channels,
                               kernel_size=final_kernel_size,
                               padding=final_kernel_size // 2)

    def forward(self, x):
        """
        Decodes a hidden state x.
        Args:
            x : tensor (B, C, W, H), hidden states
        Returns:
            out: tensor (B, C_out, 3*W, 3*H), fused hidden state
        """

        x = self.deconv(x)
        x = self.final(x)
        return x


from .model import Model


class HighResNet(Model):
    """ HRNet, a neural network for multi-frame super resolution (MFSR) by recursive fusion. """

    def __init__(self, encoder: Encoder = Encoder(), recursive_net: RecursiveNet = RecursiveNet(),
                 decoder: Decoder = Decoder()):
        """
        Args:
            config : dict, configuration file
        """

        super(HighResNet, self).__init__()
        self.encode = encoder
        self.fuse = recursive_net
        self.decode = decoder

    def forward(self, data, alphas=None):
        '''
        Super resolves a batch of low-resolution images.
        Args:
            lrs : tensor (B, L, W, H), low-resolution images
            alphas : tensor (B, L), boolean indicator (0 if padded low-res view, 1 otherwise)
        Returns:
            srs: tensor (B, C_out, W, H), super-resolved images
        '''
        lrs: torch.Tensor = data.x
        print(lrs.shape)
        max_size = torch.tensor(lrs.shape).max()
        new_lrs = torch.zeros(lrs.shape[0], lrs.shape[1], max_size, max_size, device=lrs.device, dtype=torch.float32)
        new_lrs[:, :, :lrs.shape[-2], :lrs.shape[-1]] = lrs
        lrs = new_lrs
        batch_size, seq_len, height, width = lrs.shape
        lrs = lrs.view(-1, seq_len, 1, height, width)
        alphas = torch.ones(lrs.shape[0], lrs.shape[1], device=lrs.device)
        alphas = alphas.view(-1, seq_len, 1, 1, 1)

        refs, _ = torch.median(lrs[:, :9], 1, keepdim=True)  # reference image aka anchor, shared across multiple views
        refs = refs.repeat(1, seq_len, 1, 1, 1)
        stacked_input = torch.cat([lrs, refs], 2)  # tensor (B, L, 2*C_in, W, H))

        stacked_input = stacked_input.view(batch_size * seq_len, 2, width, height)
        layer1 = self.encode(stacked_input)  # encode input tensor
        layer1 = layer1.view(batch_size, seq_len, -1, width, height)  # tensor (B, L, C, W, H)

        # fuse, upsample
        recursive_layer = self.fuse(layer1, alphas)  # fuse hidden states (B, C, W, H)
        srs = self.decode(recursive_layer)  # decode final hidden state (B, C_out, 3*W, 3*H)
        srs = torch.nn.Sigmoid()(srs)
        srs = srs[:, :, :data.x.shape[-2]*3, :data.x.shape[-1]*3]
        print(srs.shape)
        return srs


class ShiftNet(nn.Module):
    """ ShiftNet, a neural network for sub-pixel registration and interpolation with lanczos kernel. """

    def __init__(self, image_shape, out_channels: int = 64, in_channel=1):
        """
        Args:
            in_channel : int, number of input channels
        """

        super(ShiftNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(2 * in_channel, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.drop1 = nn.Dropout(p=0.5)
        final_size = image_shape // 8
        self.fc1 = nn.Linear(128 * final_size * final_size, 1024)
        self.activ1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 2, bias=False)
        self.fc2.weight.data.zero_()  # init the weights with the identity transformation

    def forward(self, x):
        """
        Registers pairs of images with sub-pixel shifts.
        Args:
            x : tensor (B, 2*C_in, H, W), input pairs of images
        Returns:
            out: tensor (B, 2), translation params
        """
        x[:, 0] = x[:, 0] - torch.mean(x[:, 0], dim=(1, 2)).view(-1, 1, 1)
        x[:, 1] = x[:, 1] - torch.mean(x[:, 1], dim=(1, 2)).view(-1, 1, 1)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.view(-1, 128 * out.shape[-2] * out.shape[-1])
        out = self.drop1(out)  # dropout on spatial tensor (C*W*H)
        out = self.fc1(out)
        out = self.activ1(out)
        out = self.fc2(out)
        return out

    def transform(self, theta, I, device="cpu"):
        """
        Shifts images I by theta with Lanczos interpolation.
        Args:
            theta : tensor (B, 2), translation params
            I : tensor (B, C_in, H, W), input images
        Returns:
            out: tensor (B, C_in, W, H), shifted images
        """

        self.theta = theta
        new_I = lanczos_shift(img=I.transpose(0, 1),
                              shift=self.theta.flip(-1),  # (dx, dy) from register_batch -> flip
                              a=3, p=5)[:, None]
        return new_I


def lanczos_kernel(dx, a=3, N=None, dtype=None, device=None):
    """
    Generates 1D Lanczos kernels for translation and interpolation.
    Args:
        dx : float, tensor (batch_size, 1), the translation in pixels to shift an image.
        a : int, number of lobes in the kernel support.
            If N is None, then the width is the kernel support (length of all lobes),
            S = 2(a + ceil(dx)) + 1.
        N : int, width of the kernel.
            If smaller than S then N is set to S.
    Returns:
        k: tensor (?, ?), lanczos kernel
    """

    if not torch.is_tensor(dx):
        dx = torch.tensor(dx, dtype=dtype, device=device)

    if device is None:
        device = dx.device

    if dtype is None:
        dtype = dx.dtype

    D = dx.abs().ceil().int()
    S = 2 * (a + D) + 1  # width of kernel support

    S_max = S.max() if hasattr(S, 'shape') else S

    if (N is None) or (N < S_max):
        N = S

    Z = (N - S) // 2  # width of zeros beyond kernel support

    start = (-(a + D + Z)).min()
    end = (a + D + Z + 1).max()
    x = torch.arange(start, end, dtype=dtype, device=device).view(1, -1) - dx
    px = (np.pi * x) + 1e-3

    sin_px = torch.sin(px)
    sin_pxa = torch.sin(px / a)

    k = a * sin_px * sin_pxa / px ** 2  # sinc(x) masked by sinc(x/a)

    return k


def lanczos_shift(img, shift, p=3, a=3):
    """
    Shifts an image by convolving it with a Lanczos kernel.
    Lanczos interpolation is an approximation to ideal sinc interpolation,
    by windowing a sinc kernel with another sinc function extending up to a
    few nunber of its lobes (typically a=3).

    Args:
        img : tensor (batch_size, channels, height, width), the images to be shifted
        shift : tensor (batch_size, 2) of translation parameters (dy, dx)
        p : int, padding width prior to convolution (default=3)
        a : int, number of lobes in the Lanczos interpolation kernel (default=3)
    Returns:
        I_s: tensor (batch_size, channels, height, width), shifted images
    """

    dtype = img.dtype
    if len(img.shape) == 2:
        img = img[None, None].repeat(1, shift.shape[0], 1, 1)  # batch of one image
    elif len(img.shape) == 3:  # one image per shift
        assert img.shape[0] == shift.shape[0]
        img = img[None,]

    # Apply padding

    padder = torch.nn.ReflectionPad2d(p)  # reflect pre-padding
    I_padded = padder(img)

    # Create 1D shifting kernels

    y_shift = shift[:, [0]]
    x_shift = shift[:, [1]]

    k_y = (lanczos_kernel(y_shift, a=a, N=None, dtype=dtype)
           .flip(1)  # flip axis of convolution
           )[:, None, :, None]  # expand dims to get shape (batch, channels, y_kernel, 1)
    k_x = (lanczos_kernel(x_shift, a=a, N=None, dtype=dtype)
           .flip(1)
           )[:, None, None, :]  # shape (batch, channels, 1, x_kernel)

    # Apply kernels

    I_s = torch.conv1d(I_padded,
                       groups=k_y.shape[0],
                       weight=k_y,
                       padding=[k_y.shape[2] // 2, 0])  # same padding
    I_s = torch.conv1d(I_s,
                       groups=k_x.shape[0],
                       weight=k_x,
                       padding=[0, k_x.shape[3] // 2])

    I_s = I_s[..., p:-p, p:-p]  # remove padding

    return I_s.squeeze()  # , k.squeeze()


def register_batch_and_apply_shifts(shift_net: ShiftNet, sr: torch.Tensor, image_size,
                                    hr: torch.Tensor, device='cpu', **kwargs):
    predicted_regis = sr[:, :, image_size // 2 - image_size // 4:image_size // 2 + image_size // 4,
                      image_size // 2 - image_size // 4:image_size // 2 + image_size // 4, ]
    reference_regis = hr[:, :, image_size // 2 - image_size // 4:image_size // 2 + image_size // 4,
                      image_size // 2 - image_size // 4:image_size // 2 + image_size // 4]
    n_views = sr.shape[1]
    thetas = []
    for i in range(n_views):
        theta = shift_net(torch.cat([reference_regis, predicted_regis[:, i: i + 1]], 1))
        thetas.append(theta)
    thetas = torch.stack(thetas, 1)

    batch_size, n_views, height, width = sr.shape
    images = sr.view(-1, 1, height, width)
    thetas = thetas.view(-1, 2)
    new_images = shift_net.transform(thetas, images, device=device)
    new_images = new_images.view(-1, n_views, images.size(2), images.size(3))
    return new_images
