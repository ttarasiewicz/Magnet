import os
from typing import Union
from .dataset_configurator import DatasetConfigurator
from sr_core.data.dataset import LazyImageDataset
from sr_core.constants import Subsets, EntryElements
import sr_core.data.transformations as dtrnf
import sr_core.image.transformations as itrnf


class S2_Configurator(DatasetConfigurator):
    def __init__(self, root, origin_dir, bands, degradation):
        if bands == '10m':
            bands = ['b2', 'b3', 'b4', 'b8']
        super().__init__(root, origin_dir)
        if not isinstance(bands, list):
            bands = [bands]
        self.used_bands = bands
        self._degradation = degradation
        self.path = os.path.join(self.root, self.name)
        assert os.path.isdir(self.path), f"No such directory: {self.path}"

    @property
    def _prefix(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._prefix + self._degradation

    @property
    def bands(self):
        return [f'b{i}' for i in range(1, 13)] + ['b8a']

    @property
    def band_resolutions(self):
        return {
            'b1': 60, 'b2': 10, 'b3': 10, 'b4': 10, 'b5': 20, 'b6': 20,
            'b7': 20, 'b8': 10, 'b8a': 20, 'b9': 60, 'b10': 60, 'b11': 20, 'b12': 20
        }


class S2_Artificial(S2_Configurator):
    origin_dir = \
        r'\\earth.kplabs.pl\pub\Teams\Projekty ML\SRR\datasets\3 DeepSent\Sentinel_2_artificial\s2_bicubic'

    def __init__(self, bands: str = None, scale: int = 3,
                 root: str = None, degradation: str = 'bicubic'):
        super(S2_Artificial, self).__init__(root, self.origin_dir, bands, degradation)
        self.scale = scale

    @property
    def _prefix(self):
        return 's2_'

    @property
    def has_masks(self):
        return False

    def _prepare_dataset(self, subset: Union[Subsets, str]):
        test_train_ratio = 0.2
        val_train_ratio = 0.2
        train_dataset = LazyImageDataset(
            self.path,
            additional_deepness=1, scale=self.scale)
        train_dataset._examples_dirs = [os.path.join(example, band) for band in self.used_bands
                                        for example in train_dataset.get_examples_dirs()]
        train_dataset = train_dataset.shuffle(0)
        all_examples = len(train_dataset)
        test_examples = int(all_examples * test_train_ratio)
        valid_examples = int((all_examples - test_examples) * val_train_ratio)
        train_examples = all_examples - test_examples - valid_examples
        test_dataset = train_dataset.take(test_examples)
        valid_dataset = train_dataset.take(valid_examples, test_examples)
        train_dataset = train_dataset.take(train_examples, test_examples + valid_examples)

        dataset = {Subsets.TRAIN.value: train_dataset,
                   Subsets.TEST.value: test_dataset,
                   Subsets.VALID.value: valid_dataset}[subset.value]
        return dataset

    def _preprocess(self, dataset):
        standardize = dtrnf.EntryTransformationBuilder() \
            .with_transformation(itrnf.Divide(2 ** 16 - 1), target=[EntryElements.LRS, EntryElements.HR]) \
            .without_rest_transformation() \
            .build()

        to_float = dtrnf.EntryTransformationBuilder() \
            .with_transformation(itrnf.ToFloat(), target=[EntryElements.LRS, EntryElements.HR]) \
            .without_rest_transformation() \
            .build()

        dataset = dataset.transform(standardize)
        dataset = dataset.transform(to_float)
        return dataset

    @property
    def raw_file_names(self):
        return None
