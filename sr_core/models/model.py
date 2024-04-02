import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return self.__class__.__name__


class GraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return self.__class__.__name__
