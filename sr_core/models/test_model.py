import torch
import torch_geometric as tg
from torch_geometric.nn import GCNConv, SplineConv, LayerNorm, graclus, max_pool_x, global_max_pool
import torch.nn.functional as F
import math
from torch_geometric.data import Data
from torch_geometric.utils import normalized_cut
from torch import nn
from torch_geometric.transforms import Cartesian


class TestModel(torch.nn.Module):

    def __init__(self, filters, scale):
        super().__init__()
        self.filters = filters
        self.scale = scale
        self.conv1 = GCNConv(9, filters)
        self.conv2 = GCNConv(filters, filters)
        self.conv3 = GCNConv(filters, filters)
        self.conv4 = GCNConv(filters, scale**2)
        self.conv5 = GCNConv(scale**2, 1)
        self.bn1 = LayerNorm(filters)
        self.bn2 = LayerNorm(filters)
        self.bn3 = LayerNorm(filters)

    def forward(self, data):
        x = data.x
        x = self.conv1(x, data.edge_index)
        x = F.elu(self.bn2(x))
        x = self.conv2(x, data.edge_index)
        x = F.elu(self.bn2(x))
        x = self.conv3(x, data.edge_index)
        x = F.elu(self.bn3(x))
        x = F.elu(self.conv4(x, data.edge_index))
        x = F.elu(self.conv5(x, data.edge_index))
        x = x.reshape((math.isqrt(x.shape[0]), math.isqrt(x.shape[0]), x.shape[-1])).permute([2, 0, 1])
        return x

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))



class SplineModel(torch.nn.Module):
    def __init__(self, hr_size):
        super().__init__()
        self.hr_size = hr_size
        self.first_pass = True
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=3)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=3)
        self.conv3 = SplineConv(64, 64, dim=2, kernel_size=3)
        self.conv4 = SplineConv(64, 64, dim=2, kernel_size=3)
        self.conv5 = SplineConv(64, 32, dim=2, kernel_size=3)
        self.conv6 = SplineConv(32, 32, dim=2, kernel_size=3)
        self.conv7 = SplineConv(32, 32, dim=2, kernel_size=3)
        self.conv8 = SplineConv(32, 32, dim=2, kernel_size=3)
        self.conv9 = SplineConv(32, 1, dim=2, kernel_size=3)
        self.transform = Cartesian(cat=False)

    def prints(self, *x):
        if self.first_pass:
            print(*x)

    def forward(self, data: Data):
        x = data.x
        self.prints('Forward', data.edge_attr.min(), data.edge_attr.max())
        self.prints(data.edge_attr.shape, data.edge_attr)
        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv4(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv5(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv6(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv7(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv8(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv9(x, data.edge_index, data.edge_attr))
        self.prints("Final", x.shape, data.y.shape)
        x = x.reshape([1, data.y.shape[-1], data.y.shape[-1]])
        self.prints("Final reshaped", x.shape)
        self.first_pass = False
        return x
