import collections
import typing

from torch.nn import ModuleList, UpsamplingBilinear2d, UpsamplingNearest2d, modules, Sigmoid
from torch.nn import modules
import torch_geometric as tg

from sr_core import utils
from sr_core.models.model import Model, GraphModel

# Description of these parameters can be found in the paper.
FSRCNNSensitiveParameters = collections.namedtuple("FSRCNNParameters",
                                                   ["feature_dimensions",
                                                    "feature_shrinking"])


def calculate_padding(kernel_size: int) -> int:
    if kernel_size % 2 == 0:
        raise RuntimeError("Padding cannot be calculated for kernel with "
                           "even size.")
    return kernel_size // 2


class FSRCNN(Model):
    """
    Fast Super-Resolution Convolutional Network based on paper titled
    "Accelerating the Super-Resolution Convolutional Neural Network" by Dong et al.

    """
    EXTRACTION_KERNEL_SIZE = 5
    SHRINKING_KERNEL_SIZE = 1
    MAPPING_KERNEL_SIZE = 3
    EXPANDING_KERNEL_SIZE = 1
    DECONVOLUTION_KERNEL_SIZE = 9

    def __init__(self, channels: int = 1,
                 upscale_factor: int = 3,
                 non_linear_mapping_layers: int = 4,
                 sensitive_parameters: FSRCNNSensitiveParameters = FSRCNNSensitiveParameters(56, 16)):
        """
        All arguments mentioned below should be positive integrals.

        :param channels: number of color channels
        :param non_linear_mapping_layers: number of non linear mapping layers
        :param upscale_factor: how many times the image size should increase
        """
        d = sensitive_parameters.feature_dimensions
        s = sensitive_parameters.feature_shrinking
        super().__init__()
        feature_extraction = modules.Sequential(
            modules.Conv2d(in_channels=channels, out_channels=d,
                           kernel_size=FSRCNN.EXTRACTION_KERNEL_SIZE,
                           padding=calculate_padding(
                               self.EXTRACTION_KERNEL_SIZE)),
            modules.PReLU())
        shrinking = modules.Sequential(
            modules.Conv2d(in_channels=d, out_channels=s,
                           kernel_size=self.SHRINKING_KERNEL_SIZE),
            modules.PReLU())
        non_linear_mapping = []
        for _ in range(non_linear_mapping_layers):
            non_linear_mapping.append(
                modules.Conv2d(in_channels=s, out_channels=s,
                               kernel_size=self.MAPPING_KERNEL_SIZE,
                               padding=calculate_padding(
                                   self.MAPPING_KERNEL_SIZE)))
            non_linear_mapping.append(modules.PReLU())

        expanding = modules.Sequential(
            modules.Conv2d(in_channels=s, out_channels=d,
                           kernel_size=self.EXPANDING_KERNEL_SIZE),
            modules.PReLU())

        pad, out_pad = _calculate_deconvolution_padding(upscale_factor)
        deconvolution = modules.ConvTranspose2d(in_channels=d,
                                                out_channels=channels,
                                                kernel_size=self.DECONVOLUTION_KERNEL_SIZE,
                                                stride=upscale_factor,
                                                padding=pad,
                                                output_padding=out_pad
                                                )

        self.full_model = modules.Sequential(feature_extraction,
                                             shrinking,
                                             *non_linear_mapping,
                                             expanding,
                                             deconvolution)

    def forward(self, x):
        x = self.full_model(x.x)
        return Sigmoid()(x)


def _calculate_deconvolution_padding(factor: int) \
        -> typing.Tuple[int, int]:
    """
    For different scale factor image must be differently padded so the
    output image shape will be scaled correctly i.e. without losing some
    pixels at borders.

    :param factor: upscale factor.
    :return: padding and output padding that must be applied at the
        deconvolution layer.
    """
    paddings = {
        2: (4, 1),
        3: (3, 0),
        4: (3, 1),
        8: (2, 3),
    }
    padding, output_padding = paddings.setdefault(factor, (3, 1))
    return padding, output_padding


class GraphFSRCNN(GraphModel):
    """
    Fast Super-Resolution Convolutional Network based on paper titled
    "Accelerating the Super-Resolution Convolutional Neural Network" by Dong et al.

    """
    EXTRACTION_KERNEL_SIZE = 5
    SHRINKING_KERNEL_SIZE = 1
    MAPPING_KERNEL_SIZE = 3
    EXPANDING_KERNEL_SIZE = 1
    DECONVOLUTION_KERNEL_SIZE = 9

    def __init__(self, channels: int = 1,
                 upscale_factor: int = 3,
                 non_linear_mapping_layers: int = 4,
                 sensitive_parameters: FSRCNNSensitiveParameters = FSRCNNSensitiveParameters(56, 16)):
        """
        All arguments mentioned below should be positive integrals.

        :param channels: number of color channels
        :param non_linear_mapping_layers: number of non linear mapping layers
        :param upscale_factor: how many times the image size should increase
        """
        d = sensitive_parameters.feature_dimensions
        s = sensitive_parameters.feature_shrinking
        super().__init__()
        self.feature_extraction = tg.nn.SplineConv(channels, d, dim=2,
                                                   kernel_size=self.EXTRACTION_KERNEL_SIZE)
        self.prelu1 = modules.PReLU()
        self.shrinking = tg.nn.SplineConv(d, s, dim=2, kernel_size=self.SHRINKING_KERNEL_SIZE)
        self.prelu2 = modules.PReLU()
        self.non_linear_mapping = []
        self.non_linear_prelus = []
        for _ in range(non_linear_mapping_layers):
            self.non_linear_mapping.append(
                tg.nn.SplineConv(s, s, 2, self.MAPPING_KERNEL_SIZE))
            self.non_linear_prelus.append(modules.PReLU())
        self.non_linear_mapping = ModuleList(self.non_linear_mapping)
        self.non_linear_prelus = ModuleList(self.non_linear_prelus)
        self.expanding = tg.nn.SplineConv(s, d, 2, self.EXPANDING_KERNEL_SIZE)
        self.prelu3 = modules.PReLU()
        pad, out_pad = _calculate_deconvolution_padding(upscale_factor)
        self.deconvolution = modules.ConvTranspose2d(in_channels=d,
                                                     out_channels=channels,
                                                     kernel_size=self.DECONVOLUTION_KERNEL_SIZE,
                                                     stride=upscale_factor,
                                                     padding=pad,
                                                     output_padding=out_pad
                                                     )

    def forward(self, batch: tg.data.Batch):
        x = batch.x
        x = self.prelu1(self.feature_extraction(x, batch.edge_index, batch.edge_attr))
        x = self.prelu2(self.shrinking(x, batch.edge_index, batch.edge_attr))
        for mapping, relu in zip(self.non_linear_mapping, self.non_linear_prelus):
            x = relu(mapping(x, batch.edge_index, batch.edge_attr))
        x = self.prelu3(self.expanding(x, batch.edge_index, batch.edge_attr))
        x = utils.graph_to_image2d(x, batch.num_graphs)
        # x = self.upsample(x)
        x = self.deconvolution(x)

        # x, edge_index, pos = utils.image2d_to_graph(x)
        # data = tg.data.Data(x, edge_index, pos=pos)
        # data = self.cartesian(data).to(x.device)


        # x = utils.graph_to_image2d(x)
        return Sigmoid()(x)
