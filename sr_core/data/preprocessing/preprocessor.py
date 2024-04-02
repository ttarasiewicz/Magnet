import torch
import torch_geometric


class Preprocessor:
    def __init__(self):
        self.operations = []

    def add_to_graph(self, dataset):
        for operation in self.operations:
            dataset = dataset.map(operation)
        return dataset
