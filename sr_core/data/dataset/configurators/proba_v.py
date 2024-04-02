import os.path as osp
from sr_core.constants import Subsets, EntryElements
from sr_core.data.dataset import StaticImageDataset, LazyImageDataset
from .dataset_configurator import DatasetConfigurator
import sr_core.data.transformations as dtrnf
import sr_core.image.transformations as itrnf
from sr_core.image.registration import get_entry_translations, register_entry


class ProbaV_RAMS(DatasetConfigurator):
    default_origin_dir = r'\\earth.kplabs.pl\pub\Teams\Projekty ML\SRR\datasets\3 DeepSent\ProbaV_RAMS'
    default_root = r'..\datasets\ProbaV_RAMS'

    def __init__(self, bands: str = None, root: str = None, origin_dir=None, **kwargs):
        origin_dir = self.default_origin_dir if origin_dir is None else origin_dir
        root = self.default_root if root is None else root
        super(ProbaV_RAMS, self).__init__(root, origin_dir)
        if not isinstance(bands, list):
            bands = [bands]
        self.used_bands = bands
        self.root = root

    @property
    def bands(self):
        return ['NIR', 'RED']

    @property
    def has_masks(self):
        return True

    @property
    def name(self):
        return 'ProbaV_RAMS'

    def _prepare_dataset(self, subset, **kwargs):
        self.subset = subset if subset is not Subsets.TEST else Subsets.VALID
        datasets = []
        for lr_file, hr_file, hr_mask_file in self.raw_file_names:
            dataset = StaticImageDataset.from_numpy_file(lr_file=lr_file,
                                                         hr_file=hr_file,
                                                         hr_masks_file=hr_mask_file,
                                                         directory=self.root,
                                                         is_NCHW=False)
            datasets.append(dataset)
        dataset = StaticImageDataset([entry for data in datasets for entry in data])
        return dataset

    def _preprocess(self, dataset, **kwargs):
        standardize = (dtrnf.EntryTransformationBuilder()
                       .with_transformation(itrnf.Multiply(1 / (2 ** 16 - 1)),
                                            target=[EntryElements.HR, EntryElements.LRS])
                       .without_rest_transformation()
                       .build())
        dataset = dataset.transform(standardize)
        return dataset

    @property
    def raw_file_names(self):
        filenames = [[f'X_{band.upper()}_{self.subset.value}.npy',
                      f'y_{band.upper()}_{self.subset.value}.npy',
                      f'y_{band.upper()}_{self.subset.value}_masks.npy'] for band in self.used_bands]
        return filenames


class ProbaV(DatasetConfigurator):
    origin_dir = ''

    def __init__(self, dataset_name: str, bands: str = None, root: str = None, **kwargs):
        super(ProbaV, self).__init__(root=root)
        self.band = bands
        self.name = dataset_name

    @property
    def has_masks(self):
        return True

    def _divide_dataset_into_subsets(self, register_lrs=False, register_mode='float', compute_shifts=True):
        val_train_ratio = 0.2
        train_dataset = LazyImageDataset(
            osp.join(self.root, self.name, 'train', self.band),
            additional_deepness=0, register_lrs=register_lrs, register_mode=register_mode,
            compute_shifts=compute_shifts)
        test_dataset = LazyImageDataset(
            osp.join(self.root, self.name, 'test', self.band),
            additional_deepness=0, register_lrs=register_lrs, register_mode=register_mode,
            compute_shifts=compute_shifts)
        train_dataset = train_dataset.shuffle(0)
        all_examples = len(train_dataset)
        valid_examples = int(all_examples * val_train_ratio)
        train_examples = all_examples - valid_examples
        valid_dataset = train_dataset.take(valid_examples)
        train_dataset = train_dataset.take(train_examples, valid_examples)
        return train_dataset, valid_dataset, test_dataset

    def get_config_dict(self):
        train, valid, test = self._divide_dataset_into_subsets()
        datasets = {}
        datasets['band'] = self.band
        datasets['train'] = [x.split('\\')[-1] for x in train.get_raw_examples()]
        datasets['val'] = [x.split('\\')[-1] for x in valid.get_raw_examples()]
        datasets['test'] = [x.split('\\')[-1] for x in test.get_raw_examples()]
        config_dict = {self.name: datasets}
        return config_dict

    def _prepare_dataset(self, subset, register_lrs=False, register_mode='float', compute_shifts=True, **kwargs):
        train_dataset, valid_dataset, test_dataset = self._divide_dataset_into_subsets(register_lrs,
                                                                                       register_mode,
                                                                                       compute_shifts=compute_shifts)
        dataset = {Subsets.TRAIN: train_dataset,
                   Subsets.TEST: test_dataset,
                   Subsets.VALID: valid_dataset}.get(subset)
        return dataset

    def _preprocess(self, dataset, **kwargs):
        standardize = dtrnf.EntryTransformationBuilder() \
            .with_transformation(itrnf.Divide(2**14-1), target=[EntryElements.LRS, EntryElements.HR]) \
            .without_rest_transformation() \
            .build()

        # standardize = dtrnf.EntryTransformationBuilder() \
        #     .with_transformation(itrnf.Normalize(), target=[EntryElements.LRS, EntryElements.HR]) \
        #     .without_rest_transformation() \
        #     .build()

        convert_masks = dtrnf.EntryTransformationBuilder() \
            .with_transformation(itrnf.Divide(255), target=[EntryElements.LR_MASKS, EntryElements.HR_MASK]) \
            .without_rest_transformation() \
            .build()

        to_float = dtrnf.EntryTransformationBuilder() \
            .with_transformation(itrnf.ToFloat(), target=[EntryElements.LRS, EntryElements.HR,
                                                          EntryElements.LR_MASKS, EntryElements.HR_MASK]) \
            .without_rest_transformation() \
            .build()

        dataset = dataset.transform(to_float)
        dataset = dataset.transform(standardize)
        dataset = dataset.transform(convert_masks)
        return dataset

    @property
    def raw_file_names(self):
        return None
