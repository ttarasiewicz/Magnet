from .model import GraphModel
import torch_geometric as tg
import torch
from torch.nn import modules, ModuleList
import torch.nn.functional as F
from sr_core import utils
from sr_core.nn import interpolation
from sr_core.data import GraphEntry
from torch_geometric.utils import normalized_cut


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class MagNet(GraphModel):
    def __init__(self, in_channels: int = 1, scale: int = 3, processing_layers=4):
        super().__init__()
        d = 34
        s = 16
        self.scale = scale
        self.feature_extraction = tg.nn.SplineConv(in_channels, d, dim=2, kernel_size=3)
        self.prelu1 = modules.PReLU()
        self.shrinking = tg.nn.SplineConv(d, s, dim=2, kernel_size=1)
        self.prelu2 = modules.PReLU()
        non_linear_mapping = []
        non_linear_prelus = []
        for _ in range(processing_layers):
            non_linear_mapping.append(
                tg.nn.SplineConv(s, s, 2, 3))
            non_linear_prelus.append(modules.PReLU())
        self.non_linear_mapping = ModuleList(non_linear_mapping)
        self.non_linear_prelus = ModuleList(non_linear_prelus)
        self.expanding = tg.nn.SplineConv(s, d, 2, 1)
        self.prelu3 = modules.PReLU()
        self.final = tg.nn.SplineConv(d, scale**2, 2, 3)
        self.prelu4 = modules.PReLU()
        pass

    def forward(self, batch: tg.data.Batch):
        num_graphs = batch.num_graphs
        x = batch.x
        image_len = x.shape[0]//num_graphs
        # batch_indices = torch.arange(num_graphs, device=x.device)
        # batch_indices = torch.repeat_interleave(batch_indices, image_len, 0)
        cluster = batch.batch*image_len
        cluster = torch.arange(image_len//9, device=x.device).repeat(9).repeat(num_graphs).add(cluster)
        x = self.prelu1(self.feature_extraction(x, batch.edge_index, batch.edge_attr))
        x_res = x
        x = self.prelu2(self.shrinking(x, batch.edge_index, batch.edge_attr))
        for mapping, relu in zip(self.non_linear_mapping, self.non_linear_prelus):
            x = relu(mapping(x, batch.edge_index, batch.edge_attr))
        x = self.prelu3(self.expanding(x, batch.edge_index, batch.edge_attr))
        x = x + x_res
        x = self.prelu4(self.final(x, batch.edge_index, batch.edge_attr))
        x, batch_indices = tg.nn.avg_pool_x(cluster, x, batch.batch)
        x = utils.graph_to_image2d(x, num_graphs)
        x = F.pixel_shuffle(x, self.scale).contiguous()
        return x


class MagNetv2(GraphModel):
    # - Reducing LRs is performed by max pooling nodes corresponding to the same pixel in tensor form.

    def __init__(self, in_channels: int = 1, scale: int = 3, processing_layers=4, lrs=9):
        super().__init__()
        d = 56
        s = 16
        self.lrs = lrs
        self.scale = scale
        self.feature_extraction = tg.nn.SplineConv(in_channels, d, dim=2, kernel_size=3)
        self.prelu1 = modules.PReLU()
        self.shrinking = tg.nn.SplineConv(d, s, dim=2, kernel_size=1)
        self.prelu2 = modules.PReLU()
        non_linear_mapping = []
        non_linear_prelus = []
        for _ in range(processing_layers):
            non_linear_mapping.append(
                tg.nn.SplineConv(s, s, 2, 3))
            non_linear_prelus.append(modules.PReLU())
        self.non_linear_mapping = ModuleList(non_linear_mapping)

        self.non_linear_prelus = ModuleList(non_linear_prelus)
        self.expanding = tg.nn.SplineConv(s, d, 2, 1)
        self.prelu3 = modules.PReLU()
        self.pre_register = tg.nn.SplineConv(d, d, 2, 3,)
        self.prelu4 = modules.PReLU()

        register = []
        register_prelus = []
        for _ in range(4):
            register.append(
                tg.nn.SplineConv(d, d, 2, 3))
            register_prelus.append(modules.PReLU())
        self.register = ModuleList(register)
        self.register_prelus = ModuleList(register_prelus)

        self.final = tg.nn.SplineConv(d, scale**2, 2, 3)
        self.prelu5 = modules.PReLU()
        self.transform = tg.transforms.Cartesian()

    def forward(self, batch: GraphEntry):
        # print(batch.lrs.min(), batch.lrs.max())
        num_graphs = batch.num_graphs
        x = batch.lrs
        image_len = x.shape[0] // num_graphs
        cluster = batch.batch * image_len
        cluster = torch.arange(image_len // self.lrs, device=x.device).repeat(self.lrs).repeat(num_graphs).add(cluster)
        x = self.prelu1(self.feature_extraction(x, batch.edge_index, batch.edge_attr))
        x_res = x
        x = self.prelu2(self.shrinking(x, batch.edge_index, batch.edge_attr))
        for mapping, prelu in zip(self.non_linear_mapping, self.non_linear_prelus):
            x = prelu(mapping(x, batch.edge_index, batch.edge_attr))
        x = self.prelu3(self.expanding(x, batch.edge_index, batch.edge_attr))
        x = x + x_res
        x, batch_indices = tg.nn.max_pool_x(cluster, x, batch.batch)
        edge_index, pos = tg.utils.grid(batch.lr_shape[0, 0], batch.lr_shape[0, 1],
                                        device=x.device)
        data = tg.data.Data(edge_index=edge_index, pos=pos)
        data = self.transform(data)
        data = tg.data.Batch.from_data_list([data]*batch.num_graphs)

        x = self.prelu4(self.pre_register(x, data.edge_index, data.edge_attr))
        x_res = x
        for register, prelu in zip(self.register, self.register_prelus):
            x = prelu(register(x, data.edge_index, data.edge_attr))
        x = x + x_res
        x = self.prelu5(self.final(x, data.edge_index, data.edge_attr))
        x = utils.graph_to_image2d(x, num_graphs,
                                   lr_shape=(batch.lr_shape[0, 0], batch.lr_shape[0, 1]))
        x = F.pixel_shuffle(x, self.scale)
        # x = torch.nn.Sigmoid()(x)
        return x