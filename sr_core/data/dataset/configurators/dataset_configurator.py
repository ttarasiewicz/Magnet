from sr_core.constants import EntryElements, Subsets
from sr_core.data.entry import Entry

class DatasetConfigurator:
    root = r'..\datasets'

    def __init__(self, register_lrs: bool = False, compute_translations: bool = True, root: str = None):
        if root:
            self.root = root
        self.register_lrs = register_lrs
        self.compute_translations = compute_translations

    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def has_masks(self):
        raise NotImplementedError

    def _check_integrity(self):
        raise NotImplementedError

    def _exclude_images(self, dataset, **kwargs):
        return dataset

    def _prepare_dataset(self, subset, **kwargs):
        raise NotImplementedError

    def _preprocess(self, dataset, **kwargs):
        return dataset

    def get_dataset(self, subset: Subsets, register_lrs=False, register_mode='float', **kwargs):
        dataset = self._prepare_dataset(subset, register_lrs=register_lrs, register_mode=register_mode, **kwargs)
        dataset = self._exclude_images(dataset, **kwargs)
        dataset = self._preprocess(dataset, **kwargs)
        return dataset


def assign_value_to_items_keys(lrs=None, hr=None, lr_masks=None, hr_mask=None) -> dict:
    result = {}
    for item, data in zip(EntryElements, [lrs, hr, lr_masks, hr_mask]):
        result[item] = data
    return result