import os
from typing import Union
from .dataset_configurator import DatasetConfigurator
from sr_core.data.dataset import LazyImageDataset, StaticImageDataset
from sr_core.constants import Subsets, EntryElements
import sr_core.data.transformations as dtrnf
import sr_core.image.transformations as itrnf
from sr_core.image.registration import get_entry_translations, register_entry


class SRR_Artificial(DatasetConfigurator):
    origin_dir = None

    def __init__(self, name: str, scale: int = 3, root: str = None, **kwargs):
        super(SRR_Artificial, self).__init__(self.origin_dir, root)
        self._name = name
        self.scale = scale
        self.path = os.path.join(self.root, self.name)
        assert os.path.isdir(self.path), f"No such directory: {self.path}"

    @property
    def name(self):
        return self._name

    @property
    def has_masks(self):
        return False

    def _prepare_dataset(self, subset: Union[Subsets, str], dataset_name=None,
                         register_lrs=False, register_mode='float', **kwargs):
        dataset = LazyImageDataset(
            os.path.join(self.path, subset.value), scale=self.scale,
            additional_deepness=1, register_lrs=register_lrs, register_mode=register_mode)
        if dataset_name not in [None, '', 'all', 'combined']:
            examples = dataset.get_raw_examples()
            dataset._examples_dirs = list(filter(lambda x: dataset_name in x, examples))
        assert len(dataset.get_raw_examples())>0, "No images loaded. Probably wrong dataset was chosen."
        return dataset

    def _preprocess(self, dataset, **kwargs):
        standardize = dtrnf.EntryTransformationBuilder() \
            .with_transformation(itrnf.Divide(255), target=[EntryElements.LRS, EntryElements.HR]) \
            .without_rest_transformation() \
            .build()

        to_float = dtrnf.EntryTransformationBuilder() \
            .with_transformation(itrnf.ToFloat(), target=[EntryElements.LRS, EntryElements.HR]) \
            .without_rest_transformation() \
            .build()

        dataset = dataset.transform(standardize)
        dataset = dataset.transform(to_float)
        return dataset
