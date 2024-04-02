from sr_core.models import GraphModel
from torch_geometric.data.dataloader import DataLoader as tgDataLoader
from torch.utils.data.dataloader import DataLoader as tDataLoader


def get_loader(model, dataset, batch_size, collate_fn=None, shuffle=True, **kwargs) -> tDataLoader:
    if isinstance(model, GraphModel):
        return tgDataLoader(dataset, batch_size, shuffle, **kwargs)
    return tDataLoader(dataset, batch_size, shuffle, collate_fn=collate_fn, **kwargs)

