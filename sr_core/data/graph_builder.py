from typing import Union, Tuple, TYPE_CHECKING
import numpy as np
from torch import Tensor

from data.entry import Entry, GraphEntry
import torch_geometric as tg
import torch
from torch_geometric import nn
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm, markers
import networkx as nx


class GraphBuilder:
    def __init__(self, scale=3, transform=None, lr_shape=None, lr_images: int = None):
        self.lr_images = lr_images
        self.scale = scale
        self.transform = tg.transforms.Cartesian(norm=True) if transform is None else transform
        if lr_shape is None:
            self.builder = self._build
            self.template = None
        else:
            self.builder = self._build_from_template
            self.template = self._template(lr_shape)
        self.item = 0

    def __call__(self, entry: 'Entry'):
        if isinstance(self.lr_images, int):
            entry.lrs = entry.lrs[:self.lr_images]
            entry.lr_translations = entry.lr_translations[:self.lr_images]
        data = self.builder(entry)
        return data

    def _build(self, entry: 'Entry'):
        if self.template is None:
            lr_images = np.stack(entry.lrs, 0)
            self._template(lr_images.shape)
        return self._build_from_template(entry)

    def _build_from_template(self, entry: 'Entry'):
        raise NotImplementedError

    def _template(self, lr_shape):
        raise NotImplementedError


class FullGridBuilder(GraphBuilder):
    def __init__(self, scale=3, transform=None, lr_shape=None):
        super().__init__(scale=scale, transform=transform, lr_shape=lr_shape)

    def _build_from_template(self, entry: 'Entry'):
        lr_images = torch.tensor(np.stack(entry.lrs, 0))
        n, w, h = lr_images.shape
        translations = torch.tensor(np.stack(entry.lr_translations, 0), dtype=torch.float32)
        translations[:, 1] = -translations[:, 1]
        translations = torch.repeat_interleave(translations, w * h, 0)
        data = GraphEntry(lrs=torch.reshape(lr_images, (n * w * h, 1)),
                          hr=torch.Tensor(entry.hr[np.newaxis, np.newaxis, ...]),
                          pos=self.template.pos + translations, edge_index=self.template.edge_index,
                          lr_translations=translations)
        data = self.transform(data)
        return data

    def _template(self, shape):
        n, w, h = shape
        edge_index, pos = tg.utils.grid(w, h, dtype=torch.float32)
        pos = pos.repeat((n, 1))
        adj = tg.utils.to_dense_adj(edge_index)
        adj = adj.repeat((1, n, n)).squeeze()
        edge_index, _ = tg.utils.dense_to_sparse(adj)
        data = GraphEntry(pos=pos, edge_index=edge_index)
        return data


class RadiusBuilder(GraphBuilder):
    def __init__(self, scale: int = 3, radius: float = .999, transform=None, lr_shape=None, lr_images=None):
        super().__init__(scale=scale, transform=transform, lr_shape=lr_shape, lr_images=lr_images)
        self.radius = radius

    def _build_from_template(self, entry: 'Entry'):
        entry.lrs = entry.lrs
        entry.lr_translations = entry.lr_translations
        lr_images = np.stack(entry.lrs, 0)
        # org_lrs = lr_images
        self._template(lr_images.shape)
        n, w, h = lr_images.shape
        lr_images = torch.reshape(torch.tensor(lr_images), (n * w * h, 1))
        translations = torch.tensor(np.stack(entry.lr_translations, 0), dtype=torch.float32)
        translations = translations[:, [1, 0]]
        translations[:, 1] = -translations[:, 1]
        pos = self.template.pos + torch.repeat_interleave(translations, w * h, 0)
        edge_index = nn.radius_graph(pos, self.radius, loop=True)
        # pos = pos*self.scale
        if entry.hr_mask is not None:
            hr_mask = torch.tensor(entry.hr_mask[np.newaxis, np.newaxis, ...])
        else:
            hr_mask = None
        hr = torch.tensor(entry.hr[np.newaxis, np.newaxis, ...])
        lr_shape = torch.tensor(entry.lrs[0].shape, dtype=torch.int16).unsqueeze(0)
        lr_count = torch.tensor(len(entry.lrs), dtype=torch.uint8).unsqueeze(0)
        data = GraphEntry(lrs=lr_images,
                          hr=hr,
                          pos=pos, edge_index=edge_index, lr_translations=translations,
                          hr_mask=hr_mask, lr_shape=lr_shape,
                          lr_images=lr_count, name=entry.name)

        data = self.transform(data)
        # distance = data.edge_attr.type(torch.float32)
        # print(distance, distance.shape, distance.min(), distance.max())
        # distance = torch.square(distance)
        # print(distance, distance.shape, distance.min(), distance.max())
        # cond = torch.BoolTensor(distance < 1.0)
        # print(cond.all(1).shape, cond.all(1), cond.all(1).sum())
        # distance = distance[cond.all(1)]
        # print(distance, distance.shape, distance.min(), distance.max())
        # quit()
        # print(x)
        # indices = torch.prod(x, dim=1)==1
        # print(indices, indices.shape, indices.sum())
        # distance = distance[indices, :]
        # print(distance, distance.shape, distance.min(), distance.max())
        # # distance = distance < 1.0
        # # print()
        # # quit()
        # print(distance, distance.shape, distance.min(), distance.max())
        # distance = torch.sum(distance, 1)
        # print(distance, distance.shape, distance.min(), distance.max())
        # distance = torch.sqrt(distance)
        # print(distance, distance.shape, distance.min(), distance.max())
        #
        # print(data)
        # print(data.pos, data.edge_attr)
        # print(data.edge_attr.min(), data.edge_attr.max())
        # print(data.contains_isolated_nodes(), data.contains_self_loops())
        # print(data.lrs.min(), data.lrs.max(), torch.mean(data.lrs))
        # visualize(data, None)
        # edge_index, pos = tg.utils.grid(w*self.scale, h*self.scale)
        # x = tg.nn.knn_interpolate(data.lrs, data.pos, pos,
        #                           batch_x=torch.zeros(data.lrs.shape[0], dtype=torch.int64),
        #                           batch_y=torch.zeros(pos.shape[0], dtype=torch.int64))
        # new_data = GraphEntry(lrs=x, hr=data.y, pos=pos, edge_index=edge_index)
        # print(data)
        # print(new_data)
        # print(new_data.lrs.min(), new_data.lrs.max(), torch.mean(new_data.lrs))
        # from sr_core import utils
        # x = utils.graph_to_image2d(x, 1)
        # plt.imshow(x.squeeze().numpy(), cmap='gray')
        # plt.show()
        # print(x.shape)
        # visualize(new_data, None)
        # quit()
        return data

    def _template(self, shape):
        n, w, h = shape
        edge_index, pos = tg.utils.grid(w, h, dtype=torch.float32)
        pos = pos.repeat((n, 1))
        data = GraphEntry(pos=pos, edge_index=edge_index)
        self.template = data


def visualize(data, org_lr=None):
    data.y = None
    plt.figure(figsize=(16, 8))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(wspace=0.025, hspace=0.05)
    ax1 = plt.subplot(gs1[0])
    plt.sca(ax1)
    plt.axis('on')
    G = tg.utils.to_networkx(data)
    node_colors = data.x[:, 0].numpy().squeeze()
    nodes = nx.draw_networkx_nodes(G, data.pos.numpy(), node_size=50.0,
                                   node_color=node_colors, cmap='gray',
                                   node_shape='s')
    # print(data.edge_index.shape)
    # edges = nx.draw_networkx_edges(G, data.pos.numpy(), arrowsize=2)
    ax1.set_aspect('equal')
    if org_lr is not None:
        plt.figure(figsize=(16, 8))
        plt.imshow(org_lr[0].squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.title("0")
        plt.figure(figsize=(16, 8))
        plt.imshow(org_lr[1].squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.title("1")
    plt.show()
    # quit()
