from sr_core.data import Entry, BatchedEntry

from typing import List
import numpy as np
import torch
from torchvision.transforms import ToTensor


def _stack(tensor_list):
    if tensor_list is None:
        return None
    else:
        return torch.stack(tensor_list)


class SingleImageCollation:
    """
    Collates entries into batches of tensors. This takes only first
    lr image from the list of lr images that are in the example.

    """

    def __init__(self, image_index: int = 0):
        self._image_index = image_index

    def __call__(self, batch: List[Entry]):
        lr_batch = [ToTensor()(np.stack(entry.lrs[self._image_index:self._image_index+1], 2)) for entry in batch]
        hr_batch = [ToTensor()(entry.hr) for entry in batch]
        lr_masks = [ToTensor()(entry.lr_masks[self._image_index:self._image_index+1])
                    for entry in batch] if batch[0].lr_masks[0] is not None else None
        hr_masks = [ToTensor()(entry.hr_mask)
                    for entry in batch] if batch[0].hr_mask is not None else None
        translations = [torch.tensor(np.array(entry.lr_translations[self._image_index:self._image_index+1]))
                        for entry in batch]

        return BatchedEntry(_stack(lr_batch), _stack(hr_batch), _stack(lr_masks),
                            _stack(hr_masks), _stack(translations), name=[x.name for x in batch])


class MultipleImageCollation:
    """
    Collates entries into batches of tensors, stacking lr images to form a higher-dimensionality tensor.
    """
    def __init__(self, num_lrs=None, permute=True):
        self.num_lrs = num_lrs
        self.permute = permute

    def __call__(self, batch: List[Entry]):
        if self.permute:
            permutations = [np.random.permutation(len(entry.lrs)) for entry in batch]
        else:
            permutations = [np.arange(len(batch[0].lrs))]*len(batch)
        lr_batch = [(np.stack(entry.lrs, 2))[:, :, perm] for entry, perm in zip(batch, permutations)]
        lr_batch = [ToTensor()(lrs) for lrs in lr_batch]
        hr_batch = [ToTensor()(entry.hr) for entry in batch]

        lr_masks = [ToTensor()(np.stack(entry.lr_masks, 2)[:, :, perm])
                    for entry, perm in zip(batch, permutations)] if batch[0].lr_masks is not None else None
        hr_masks = [ToTensor()(entry.hr_mask)
                    for entry in batch] if batch[0].hr_mask is not None else None
        translations = [torch.tensor(np.array(entry.lr_translations[:self.num_lrs])) for entry in batch]
        return BatchedEntry(_stack(lr_batch), _stack(hr_batch), _stack(lr_masks),
                            _stack(hr_masks), _stack(translations), name=[x.name for x in batch])



