"""
Image resolution manipulation functions.

"""
import enum
import cv2
import numpy as np


class Filter(enum.IntEnum):
    """
    Wrapper around opencv-python interpolation methods that are described in library as variables storing ints.

    """
    NEAREST = cv2.INTER_NEAREST
    BICUBIC = cv2.INTER_CUBIC
    BILINEAR = cv2.INTER_LINEAR
    LANCZOS = cv2.INTER_LANCZOS4


def scale_image_resolution(img: np.ndarray, scaling_factor: float,
                           scaling_method: Filter = Filter.BICUBIC) -> np.ndarray:
    """
    Scales image by the scaling factor.

    :note: If scaling would result in fractions then
        it gets rounded to the nearest integer.

    :param img: Image to scale.
    :param scaling_factor: Should be non negative.
    :param scaling_method: How the scaling should be performed.
    :return: Scaled image.
    :raises ValueError: When scaling factor is not positive number.
    """
    if scaling_factor <= 0.0:
        raise ValueError(f"Scale must be a positive number not '{scaling_factor}'")
    rows = int(round(img.shape[0] * scaling_factor))
    cols = int(round(img.shape[1] * scaling_factor))
    scaled_image = cv2.resize(img, (cols, rows), interpolation=scaling_method)
    return scaled_image


def resize_image(image: np.ndarray, width: int, height: int,
                 resampling_method: Filter = Filter.BICUBIC) -> np.ndarray:
    """
    Changes the size of an image using some interpolation

    :param image: Image to resize.
    :param width: New width.
    :param height: New height.
    :param resampling_method: How to manage possible interpolation.
    :raises ValueError: When width or height is not positive number.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be a positive number not width={width} height={height}")
    resized_image = cv2.resize(image, (width, height), interpolation=resampling_method)
    return resized_image
