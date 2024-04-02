from torch_geometric.utils import grid
from torch_geometric.nn import radius
from torch_geometric.data import Data, Batch
import torch
from torch_scatter import scatter_add


def radius_reduce_interpolation(data, x, r=1.0, device='cpu'):
    edge_index, pos = grid(data.lr_shape[0, 0], data.lr_shape[0, 1], device=device)
    y_data = Data(edge_index=edge_index, pos=pos)
    y_data = Batch.from_data_list([y_data]*data.num_graphs)
    assign_index = radius(data.pos, y_data.pos, r, data.batch, y_data.batch)

    y_idx, x_idx = assign_index
    diff = data.pos[x_idx] - y_data.pos[y_idx]
    distance = (diff * diff).sum(dim=-1, keepdim=True).sqrt()
    weights = r - distance

    y = scatter_add(x[x_idx] * weights, y_idx, dim=0, dim_size=y_data.pos.size(0))
    y = y / scatter_add(weights, y_idx, dim=0, dim_size=y_data.pos.size(0))
    y_data.lrs = y
    y_data.to(device)
    return y_data
