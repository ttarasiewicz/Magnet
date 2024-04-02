"""
Module with transformations over super-resolution entries aka :class:`SREntry`.
Super-resolution entry is a high resolution image with its corresponding one or more low resolution images.


"""
import abc
import enum
import numpy as np

from typing import Callable, Union, Iterable
from sr_core.constants import EntryElements
from sr_core.data import Entry

DEFAULT_REGISTRATION_PARAMETERS = {
    "registration.algorithm": "ecc",
    "registration.ecc.numberOfIterations": "50",
    "registration.ecc.stopEpsilon": "1.0E-12",
    "registration.ecc.warpMode": "0"
}

ImageTransformation = Callable[[np.ndarray], np.ndarray]


class EntryTransformation(abc.ABC):
    """
    Base interface for defining all transformations of SREntry objects.
    """

    @abc.abstractmethod
    def __call__(self, entry: Entry) -> Entry:
        pass


class EntryTransformationBuilder:
    class SREntryTransformationBuildError(Exception):
        class Reason(enum.Enum):
            HR_TRANFORMATION_NOT_SET = "Transformation for HR image was not set."
            LR_TRANSFORMATION_NOT_SET = "Transformation for LR images was not set."
            TRANSFORMATIONS_NOT_SET = "Transformations were not set."

        def __init__(self, reason: Reason):
            self._reason = reason

        def __str__(self):
            return self._reason.value

    """
    Fluent builder for defining transformations of SREntry based dataset.
    Allows for defining transformations of LR and HR images separately.
    While describing transformations both HR and LR should be declared, i.e. if not transforming HR
    "without_hr_transformation()" must be explicitely called.
    """

    def __init__(self):
        self._transformations = dict.fromkeys(list(EntryElements))

    def with_transformation(self, transformation: ImageTransformation,
                            target: Union[EntryElements, Iterable[EntryElements]] = None)\
            -> "EntryTransformationBuilder":
        if target is None:
            target = list(EntryElements)
        if not isinstance(target, Iterable):
            target = [target]
        for key in target:
            self._transformations[key] = transformation
        return self

    def without_transformation(self,
                               target: Union[EntryElements, Iterable[EntryElements]]
                               ) -> "EntryTransformationBuilder":
        if not isinstance(target, Iterable):
            target = [target]
        for key in target:
            self._transformations[key] = lambda x: x
        return self

    def without_rest_transformation(self) -> "EntryTransformationBuilder":
        for key in self._transformations.keys():
            if self._transformations[key] is None:
                self._transformations[key] = lambda x: x
        return self

    def build(self) -> Callable:
        if not any(list(self._transformations.values())):
            raise EntryTransformationBuilder.SREntryTransformationBuildError(
                EntryTransformationBuilder.SREntryTransformationBuildError.Reason.TRANSFORMATIONS_NOT_SET)

        def transformation(entry: Entry) -> Entry:
            new_lrs = [self._transformations[EntryElements.LRS](lr) for lr in entry.lrs]
            new_hr = self._transformations[EntryElements.HR](entry.hr)
            new_lr_masks = entry.lr_masks
            new_hr_mask = entry.hr_mask
            if entry.lr_masks is not None:
                new_lr_masks = [self._transformations[EntryElements.LR_MASKS](mask) for mask in entry.lr_masks]
            if entry.hr_mask is not None:
                new_hr_mask = self._transformations[EntryElements.HR_MASK](entry.hr_mask)
            return Entry(new_lrs, new_hr, new_lr_masks, new_hr_mask, entry.lr_translations, name=entry.name)

        return transformation


# if os.name == 'nt':
#     class SREntryRegisterLRs:
#         def __init__(self, registration_parameters: Dict[str, str] = DEFAULT_REGISTRATION_PARAMETERS):
#             self.parameters = registration_parameters
#
#         def __call__(self, example: Entry) -> Entry:
#             reference = example.lrs[0]
#             for target in example.lrs[1:]:
#                 if reference.shape != target.shape:
#                     raise ValueError("Shapes mismatch between reference and target images.")
#                 registration.align_image(reference, target, parameters=self.parameters)
#             return SREntry(example.hr_image, example.lr_images)
#
#
#     class SREntryRegisterLRsToHR:
#         def __init__(self, registration_parameters: Dict[str, str] = DEFAULT_REGISTRATION_PARAMETERS):
#             self.parameters = registration_parameters
#
#         def __call__(self, example: SREntry) -> SREntry:
#             reference = example.hr_image
#             for target in example.lr_images:
#                 if reference.shape != target.shape:
#                     raise ValueError("Shapes mismatch between reference and target images.")
#                 registration.align_image(reference, target, parameters=self.parameters)
#             return SREntry(example.hr_image, example.lr_images)
