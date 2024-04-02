from torch import Tensor
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List


class Entry:
    def __init__(self, lrs, hr, lr_masks=None, hr_mask=None, lr_translations=None, name=None):
        self.lrs = lrs
        self.hr = hr
        self.lr_masks = lr_masks
        self.hr_mask = hr_mask
        self.lr_translations = lr_translations #if lr_translations is not None else [np.zeros(2)]*len(lrs)
        self.name = name

    @property
    def _items(self):
        return {"lrs": self.lrs,
                "hr": self.hr,
                'lr_masks': self.lr_masks,
                'hr_mask': self.hr_mask,
                "lr_translations": self.lr_translations,
                'name': self.name
                }

    def __getitem__(self, item):
        assert isinstance(item, str), f"Only string keys are acceptable, not {type(item)}."
        return self._items[item]

    def to(self, device):
        def send(data):
            if data is not None:
                data = data.to(device)
            return data

        self.lrs = send(self.lrs)
        self.hr = send(self.hr)
        self.lr_masks = send(self.lr_masks)
        self.hr_mask = send(self.hr_mask)
        self.lr_translations = send(self.lr_translations)
        return self


class BatchedEntry(Entry):
    def __init__(self, lrs: Tensor, hr: Tensor, lr_masks: Tensor = None,
                 hr_mask: Tensor = None,lr_translations: Tensor = None, name: List[str] = None):
        super().__init__(lrs, hr, lr_masks, hr_mask, lr_translations, name)


class GraphEntry(Data):
    # __slots__ = ["hr", "lrs", "hr_mask", 'lr_masks', 'name', 'lr_translations']

    def __init__(self, lrs=None, hr=None, lr_masks=None, hr_mask=None, lr_translations=None, edge_index=None,
                 edge_attr=None, pos=None, face=None, name=None, **kwargs):
        super().__init__(lrs=lrs, hr=hr, edge_index=edge_index, edge_attr=edge_attr,
                         pos=pos, face=face, lr_masks=lr_masks, hr_mask=hr_mask,
                         lr_translations=lr_translations, name=name, **kwargs)
