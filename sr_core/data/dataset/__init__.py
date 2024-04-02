import os
from pathlib import Path
import numpy as np
import numbers
import warnings
import copy
from typing import Iterable, Tuple, Iterator, Union, Callable, Sequence, List
import cv2
import json
from torch.utils.data import Dataset as TorchDataset
import torchdatasets as td
from sr_core.constants import Order
from sr_core import data
from sr_core.data import Entry, transformations
from sr_core.data.graph_builder import GraphBuilder
from sr_core.utils.filesystem import DirectoryContents
from sr_core.image.utils import slide_window_over_image
from sr_core.image import io
from sr_core.image.registration import get_entry_translations, register_entry


class Dataset(td.Dataset):
    """
    data.Dataset with type hints for :class:`SREntry`.
    Allows for operations like:
     - concatenating dataset (__add__)
     - picking a subset of random elements of the dataset (pick_randomly)
     - taking sublist of elements of the dataset (take)
     - reversing, shuffling, sorting and filtering the dataset

    """

    def __init__(self):
        super().__init__()

    def __getitem__(self, index) -> Entry:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def transform(self, *transformations: transformations.EntryTransformation) -> "Dataset":
        raise NotImplementedError

    def take(self, amount: int, offset: int = 0) -> "Dataset":
        max_amount = len(self)
        if amount + offset > max_amount:
            warnings.warn(f"Amount of samples to take ({amount}) is bigger than dataset "
                          f"itself. Using all elements of dataset ({max_amount}).")
            amount = max_amount
        return SubsetDataset(self, list(range(offset, amount + offset)))

    def shuffle(self, seed: int = None) -> "Dataset":
        indices = list(range(len(self)))
        np.random.seed(seed)
        np.random.shuffle(indices)
        np.random.seed(None)
        return SubsetDataset(self, indices)

    def reverse(self) -> "Dataset":
        indices = list(reversed(range(len(self))))
        return SubsetDataset(self, indices)

    def filter(self, predicate: Callable[[Entry], bool]) -> "Dataset":
        indices = []
        for entry_index, el in enumerate(self):
            if predicate(el):
                indices.append(entry_index)
        return SubsetDataset(self, indices)

    def sorted(self, key: Callable[[Entry], numbers.Real], order: Order = Order.DESC) -> "SubsetDataset":
        values = [key(entry) for entry in self]
        best_indices = np.argsort(values)
        if order is Order.DESC:
            best_indices = np.flip(best_indices)
        return SubsetDataset(self, list(best_indices))


class SubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Iterable[numbers.Integral]):
        super().__init__()
        self._dataset = dataset
        self._indices = list(indices)
        if len(self._indices) > len(self._dataset):
            raise ValueError(f"Too many indices. Subset cannot be greater than original dataset.")
        self._transformations = []

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, item) -> Entry:
        index = self._indices[item]
        entry = self._dataset[index]
        for transformation in self._transformations:
            entry = transformation(entry)
        return entry

    def transform(self, *transformations: transformations.EntryTransformation) -> "Dataset":
        copied = copy.deepcopy(self)
        copied._transformations += transformations
        return copied


class StaticImageDataset(Dataset):
    """
    This dataset is a subscriptable sequence of examples.
    Static means that the whole dataset is read to the dynamic memory at once.

    """

    def __init__(self, examples: Sequence[Entry]):
        super().__init__()
        self._examples = list(examples)

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item) -> Entry:
        return self._examples[item]

    def transform(self, *transformations: transformations.EntryTransformation) -> "StaticImageDataset":
        transformed_examples = self._examples
        for transformation in transformations:
            transformed_examples = map(transformation, transformed_examples)
        return StaticImageDataset(transformed_examples)

    @classmethod
    def from_directory(cls, directory_path: str, **kwargs):
        return cls(list(LazyImageDataset(directory_path, **kwargs)))

    @classmethod
    def from_lazy_dataset(cls, dataset: "LazyImageDataset") -> "StaticImageDataset":
        return cls(list(dataset))

    @classmethod
    def from_numpy_file(cls, lr_file: str, hr_file: str, lr_masks_file: str = None,
                        hr_masks_file: str = None, directory: str = None, is_NCHW: bool = True):
        def load_data(filename):
            if filename is None:
                return None
            if directory is not None:
                filename = os.path.join(directory, filename)
            data = list(np.load(filename).astype(np.float32))
            if not is_NCHW:
                data = [np.moveaxis(x, -1, 0) for x in data]
            data = [x.squeeze() for x in data]
            data = [list(x) if x.ndim == 3 else x for x in data]
            return data

        def check_masks(file, expected_length):
            mask_data = load_data(file)
            if mask_data is None:
                mask_data = [None for _ in range(expected_length)]
            assert len(mask_data) == expected_length, \
                f"Different number of examples for LR/HR images ({expected_length})" \
                f"and masks ({len(mask_data)}."
            return mask_data

        lr_data = load_data(lr_file)
        hr_data = load_data(hr_file)
        files_count = len(lr_data)
        assert files_count == len(hr_data), f'Different number of examples for LR images ({files_count}) and' \
                                            f'HR images ({len(hr_data)}).'
        lr_mask_data = check_masks(lr_masks_file, files_count)
        hr_mask_data = check_masks(hr_masks_file, files_count)
        dataset = [Entry(lrs, hr, lr_masks, hr_mask)
                   for hr, lrs, hr_mask, lr_masks
                   in zip(hr_data, lr_data, hr_mask_data, lr_mask_data)]
        return cls(dataset)


class PatchConfig:
    def __init__(self, shape: Tuple[int, ...], stride: Tuple[int, ...] = None):
        self.shape = shape
        self.stride = shape if stride is None else stride


# class PatchedDataset:
#     def __init__(self, hr_config: PatchConfig, lr_config: PatchConfig):
#         self.hr_config = hr_config
#         self.lr_config = lr_config
#
#     def __call__(self, dataset: Dataset) -> Iterator[data.Entry]:
#         for entry in dataset:
#             new_entries = self._crop(entry)
#             for new_entry in new_entries:
#                 yield new_entry
#
#     def _crop(self, entry: data.Entry) -> Iterator[data.Entry]:
#         hr_patches = list(slide_window_over_image(entry.hr, self.hr_config.shape,
#                                                   self.hr_config.stride))
#         lr_patch_groups = [
#             list(slide_window_over_image(lr_image, self.lr_config.shape, self.lr_config.stride)) for
#             lr_image in
#             entry.lrs]
#
#         if entry.hr_mask is not None:
#             hr_mask_patches = list(slide_window_over_image(entry.hr_mask, self.hr_config.shape,
#                                                            self.hr_config.stride))
#         else:
#             hr_mask_patches = [None] * len(hr_patches)
#         if entry.lr_masks is not None:
#             lr_masks_patch_groups = [
#                 list(slide_window_over_image(lr_image, self.lr_config.shape, self.lr_config.stride)) for
#                 lr_image in
#                 entry.lr_masks]
#         else:
#             return (data.Entry([lr_patches[i] for lr_patches in lr_patch_groups], hr_patch,
#                                None, hr_mask_patches[i],
#                                lr_translations=entry.lr_translations, name=entry.name)
#                     for i, hr_patch in enumerate(hr_patches))
#         return (data.Entry([lr_patches[i] for lr_patches in lr_patch_groups], hr_patch,
#                            [lr_masks_patches[i] for lr_masks_patches in lr_masks_patch_groups], hr_mask_patches[i],
#                            lr_translations=entry.lr_translations, name=entry.name)
#                 for i, hr_patch in enumerate(hr_patches))


class PatchedDataset(Dataset):
    def __init__(self, hr_config: PatchConfig, lr_config: PatchConfig = None, same_shapes: bool = True):
        super().__init__()
        self.hr_config = hr_config
        self.lr_config = lr_config
        self._patches_per_image = None
        self.dataset = None
        self.same_shapes = same_shapes

    def __call__(self, dataset: Dataset):  # -> Iterator[Union[SREntry, MultiBandSREntry]]:
        from tqdm import tqdm
        self.dataset = dataset
        if self._patches_per_image is None:
            if self.same_shapes:
                patches = self.get_patches(dataset[0], index_only=True)
                self._patches_per_image = [len(patches)] * len(dataset)
                return self
            self._patches_per_image = [len(self.get_patches(entry, index_only=True)) for entry in tqdm(dataset)]
        return self

    def __len__(self):
        return sum(self._patches_per_image)

    def __getitem__(self, item):
        cumsum = 0
        entry_index = None
        patch_index = None
        for i, x in enumerate(self._patches_per_image):
            if cumsum + x > item:
                entry_index = i
                patch_index = item - cumsum
                break
            cumsum += x
        entry = self.dataset[entry_index]
        patch = self.get_patches(entry)[patch_index]
        return patch

    def get_patches(self, entry: Entry, index_only: bool = False) \
            -> Sequence[Union[Entry, int]]:
        if isinstance(entry, Entry):
            patches = [x for x in self._crop(entry, index_only=index_only)]
            return patches
        else:
            raise TypeError('Entry has to be of type "SREntry" or "MultiBandSREntry".')

    def _crop(self, entry: Entry, index_only: bool = False) \
            -> List[Union[Entry, int]]:
        lr_config = self.lr_config
        hr_config = self.hr_config
        if lr_config is None:
            hr_shape = entry.hr.shape
            lr_shape = entry.lrs[0].shape
            shape_ratio = tuple(map(lambda x, y: x / y, hr_shape, lr_shape))

            lr_config = PatchConfig(
                tuple(map(lambda x, y: round(x / y), self.hr_config.shape, shape_ratio)),
                tuple(map(lambda x, y: round(x / y), self.hr_config.stride, shape_ratio))
            )
        hr_shape = round(hr_config.shape[0]), round(hr_config.shape[1])
        hr_stride = round(hr_config.stride[0]), round(hr_config.stride[1])
        lr_shape = round(lr_config.shape[0]), round(lr_config.shape[1])
        lr_stride = round(lr_config.stride[0]), round(lr_config.stride[1])
        assert hr_shape[0] == hr_config.shape[0] and hr_shape[1] == hr_config.shape[1] and \
               hr_stride[0] == hr_config.stride[0] and hr_stride[1] == hr_config.stride[1] \
               and lr_shape[0] == lr_config.shape[0] and lr_shape[1] == lr_config.shape[1] and \
               lr_stride[0] == lr_config.stride[0] and lr_stride[1] == lr_config.stride[1], \
            "Shape mismatch when patching multiple bands! Please assure that config is divisible by all coefficients: " \
            f"HR shape: {hr_shape}, LR shape: {lr_shape}"
        if index_only:
            import itertools
            starting_rows_range = range(0, entry.hr.shape[0], hr_stride[0])
            starting_columns_range = range(0, entry.hr.shape[1], hr_stride[1])
            starting_row_indices = itertools.takewhile(
                lambda index: index + hr_shape[0] <= entry.hr.shape[0],
                starting_rows_range
            )
            starting_column_indices = itertools.takewhile(
                lambda index: index + hr_shape[1] <= entry.hr.shape[1],
                starting_columns_range
            )
            return list(itertools.product(starting_row_indices, starting_column_indices))
        hr_patches = list(slide_window_over_image(entry.hr, hr_shape, hr_stride))

        lr_patch_groups = [
            list(slide_window_over_image(lr_image, lr_shape, lr_stride)) for
            lr_image in
            entry.lrs]

        lr_masks_patch_groups = [[None] * len(hr_patches) for _ in range(len(lr_patch_groups[0]))]
        hr_mask_patches = [None] * len(hr_patches)
        if entry.hr_mask is not None:
            hr_mask_patches = list(slide_window_over_image(entry.hr_mask, hr_shape,
                                                           hr_stride))
        if entry.lr_masks is not None:
            lr_masks_patch_groups = [
                list(slide_window_over_image(lr_image, lr_shape, lr_stride)) for
                lr_image in entry.lr_masks]
        return [Entry([lr_patches[i] for lr_patches in lr_patch_groups], hr_patch,
                      [lr_masks_patches[i] for lr_masks_patches in lr_masks_patch_groups], hr_mask_patches[i],
                      lr_translations=entry.lr_translations, name=f'{entry.name}_{str(i).zfill(2)}')
                for i, hr_patch in enumerate(hr_patches)]


class SISRDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, item):
        entry = self.dataset[item]
        if isinstance(entry, list):
            entry = entry[0]
        h, w = entry.lrs[0].shape
        entry.lrs = [cv2.resize(entry.hr, (w, h), interpolation=cv2.INTER_CUBIC)]
        entry.lr_translations = [np.array([0., 0.], dtype=np.float32)]
        return entry

    def __len__(self):
        return len(self.dataset)


class LazyImageDataset(Dataset):
    """
    Reads examples from directory on the fly.
    Example of directory structure that must be preserved if deepness is not set.
    ::
        |dataset
        +---img1
        |   |   hr.png
        |   |
        |   |---lr
        |   |        lr1.png
        |   |        lr2.png
        |   |        lr3.png
        |   |
        |   |---masks
        |       |    hr.png
        |       |
        |       |--- lr
        |               lr1.png
        |               lr2.png
        |               lr3.png
        +---img2
        |   |   hr.png
        |   |
        |   |---lr
        |   |        lr1.png
        |   |        lr2.png
        |   |        lr3.png
        |   |
        |   |---masks
        |       |    hr.png
        |       |
        |       |--- lr
        |               lr1.png
        |               lr2.png
        |               lr3.png

    """

    def __init__(self, directory: str, additional_deepness: int = 0, allowed_dirs: list = None,
                 scale: int = None, register_lrs=False, register_mode='float', compute_shifts=False):
        super().__init__()
        self._directory = directory
        self.scale = scale
        self.allowed_dirs = allowed_dirs if allowed_dirs is not None else []
        self._examples_dirs = [sb.path for sb in DirectoryContents.deep_list_subdirs(directory,
                                                                                     deepness=additional_deepness,
                                                                                     allowed_dirs=self.allowed_dirs)]
        self.register_mode = register_mode
        self.register_lrs = register_lrs
        self.compute_shifts = compute_shifts
        self._transformations = []

    def __len__(self):
        return len(self._examples_dirs)

    def get_raw_examples(self):
        return self._examples_dirs

    def __getitem__(self, item) -> Entry:
        example_path = self._examples_dirs[item]
        lr_subdir_entries = DirectoryContents.list_subdirs(example_path)
        lr_subdir_entries = list(filter(lambda x: 'lr' in x.name, lr_subdir_entries))
        if self.scale is not None:
            lr_subdir_entries = list(filter(lambda x: str(self.scale) + 'x' in x.name, lr_subdir_entries))
        hrs_entries = DirectoryContents.list_images_only(example_path)
        assert (len(lr_subdir_entries) == 1 and len(hrs_entries) == 1), \
            f"Scene directory ({example_path}) should contain: 1 HR image and 1 directory with LR images."

        hr_image = io.read_image(hrs_entries[0].path)
        lr_images = [io.read_image(lr_entry.path) for lr_entry in
                     (DirectoryContents.list_images_only(lr_subdir_entries[0].path))]
        assert (len(lr_images) > 0), f"At least one LR image must be present in ({example_path})."

        masks_path = os.path.join(example_path, 'masks')
        hr_mask, lr_masks, translations = None, None, None
        if os.path.isdir(masks_path):
            hr_mask_entry = DirectoryContents.list_images_only(masks_path)
            if len(hr_mask_entry) > 0:
                hr_mask = io.read_image(hr_mask_entry[0].path)
            lr_masks_subdir = DirectoryContents.list_subdirs(masks_path, allowed_dirs='lrs')
            if len(lr_masks_subdir) > 0:
                lr_masks = [io.read_image(lr_mask_entry.path) for lr_mask_entry in
                            (DirectoryContents.list_images_only(lr_masks_subdir[0].path))]
        entry = Entry(lr_images, hr_image, hr_mask=hr_mask, lr_masks=lr_masks, name=example_path.split('\\')[-1],
                      lr_translations=translations)
        if self.compute_shifts:
            entry = get_entry_translations(entry, self.register_mode)
        if self.register_lrs:
            entry = register_entry(entry, mode=self.register_mode)
        for transformation in self._transformations:
            entry = transformation(entry)
        return entry

    def transform(self, *transformations: transformations.EntryTransformation) -> "LazyImageDataset":
        copied = copy.copy(self)
        copied._transformations = copy.copy(self._transformations)
        copied._transformations += transformations
        return copied


class GraphDataset(Dataset):
    def __init__(self, dataset: Union[Iterable, Dataset], builder: GraphBuilder):
        super().__init__()
        self.builder = builder
        self.dataset = dataset

    def __getitem__(self, index) -> data.GraphEntry:
        example = self.dataset[index]
        example = self.builder(example)
        return example

    def __len__(self):
        return len(self.dataset)
