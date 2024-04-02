import torch_geometric as tg
import torch


def graph_to_image2d(nodes, num_graphs, lr_shape=None):
    length, features = nodes.shape
    length = length//num_graphs
    if lr_shape is None:
        image_size = [int(length**0.5)]*2
    else:
        image_size = lr_shape[-2:]
    nodes = nodes.reshape(-1, image_size[0], image_size[1], features)
    nodes = torch.movedim(nodes, -1, 1)
    return nodes


def image2d_to_graph(image):
    batch, features, width, height = image.shape
    image = image.view(features, width*height)
    image = torch.movedim(image, -1, 0)
    edge_index, pos = tg.utils.grid(width, height)
    data = tg.data.Data(lrs=image, edge_index=edge_index, pos=pos)
    data.to(image.device)
    return data
