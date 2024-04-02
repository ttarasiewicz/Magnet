"""Utility functions for image package."""
import itertools
import numpy as np
import skimage
from math import log, ceil
from typing import Tuple, Generator, Union, Dict, List


def restore_image_dimensions(image: np.ndarray) -> np.ndarray:
    """
    If image has 3 dimensions but only one channel it is necessary to make it
    2 dimensional to be used by conventional image processing tools.
    In short, image with only one color channel should have only 2 dimensions.

    :param image: Image.
    :return: Reshaped image if condition is fulfilled (image has 3 dimensions
        and 1 channel). Unchanged image otherwise.

    """
    channel_index = 2
    if image.ndim == 3:
        if image.shape[channel_index] == 1:
            rows, cols, _ = image.shape
            image = image.reshape((rows, cols))
    return image


def convert_float_to_byte_image(image: np.ndarray) -> np.ndarray:
    converted_image = image * 255.0
    converted_image = converted_image.clip(0, 255)
    converted_image = np.uint8(converted_image)
    return converted_image


def change_image_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Changes underlying representation of an image accroding to conventions eg.
        when converting to float values are squeezed between 0.0 to 1.0.

    :param image: Image to alter.
    :param dtype: What type to switch to.
    :return: Image with different dtype.
    """
    conversions = {
        np.uint8: skimage.img_as_ubyte,
        np.uint16: skimage.img_as_uint,
        np.float32: skimage.img_as_float32,
        np.float64: skimage.img_as_float64,
        np.bool: skimage.img_as_bool,
    }
    return conversions[dtype](image)


_TYPES_AND_RANGES: Dict[str, Tuple[Union[float, int], Union[float, int]]] = {
    "uint8": (0, 255),
    "uint16": (0, 65535),
    "float": (0.0, 1.0),
    "float16": (0.0, 1.0),
    "float32": (0.0, 1.0),
    "float64": (0.0, 1.0),
}


def get_values_range_for_dtype(dtype: np.dtype) -> Tuple[Union[float, int], Union[float, int]]:
    """
    Gets range off values for a provided dtype.

    :param dtype: numpy data type.
    :return: Tuple representing min and max values.
    :raise ValueError: When provided dtype cannot be handled.
    """
    dtype_range = _TYPES_AND_RANGES.get(dtype.name, None)
    if dtype_range is None:
        raise ValueError(f"{dtype} is a type that cannot be handled.")
    return dtype_range


def slide_window_over_image(image: np.ndarray, shape: Tuple[int, int], stride: Tuple[int, int]) -> \
        Generator[np.ndarray, None, None]:
    """
    Slides window with a given shape and stride over an image. Returns each one.

    :param image: Image to slide over.
    :param shape: Windows to use.
    :param stride: Stride to use (step in y and x direction).
    :return: Generator of each possible window.
    """
    starting_rows_range = range(0, image.shape[0], stride[0])
    starting_columns_range = range(0, image.shape[1], stride[1])
    starting_row_indices = itertools.takewhile(
        lambda index: index + shape[0] <= image.shape[0],
        starting_rows_range
    )
    starting_column_indices = itertools.takewhile(
        lambda index: index + shape[1] <= image.shape[1],
        starting_columns_range
    )
    for i, j in itertools.product(starting_row_indices,
                                  starting_column_indices):
        yield image[i:i + shape[0], j:j + shape[1]]


def split_arr_4_chunks(image: np.ndarray) -> List[np.ndarray]:
    """
    Splits image into 4 chunks

    :param image: input image
    :return: list of subimages, starting from left-bottom fragment, then left-top, right-top and right-bottom
    """
    x_res = image.shape[0]
    y_res = image.shape[1]
    middle_x = x_res // 2
    middle_y = y_res // 2
    left_arr = image[:middle_x]
    right_arr = image[middle_x:]
    left_bottom_arr = left_arr[:, :middle_y]
    left_top_arr = left_arr[:, middle_y:]
    right_top_arr = right_arr[:, middle_y:]
    right_bottom_arr = right_arr[:, :middle_y]
    return [left_bottom_arr, left_top_arr, right_top_arr, right_bottom_arr]


def split_more_times(image: np.ndarray, required_tile_shape: Tuple[int]) -> List[np.ndarray]:
    """
    Splits image into multiple of 4 chunks. Image is splitted as many times as to meet the required tile shape.

    :param image: input image
    :param required_tile_shape: required tile shape
    :return: list of subimages, starting from left-bottom fragment, then left-top, right-top and right-bottom.
    :raise: ValueError
    """

    largest_dimension = max(image.shape)
    largest_dimension_index = np.argmax(image.shape)
    required_shape_in_corresponding_max_dim = required_tile_shape[largest_dimension_index]
    if (required_shape_in_corresponding_max_dim >= largest_dimension or required_shape_in_corresponding_max_dim <= 0):
        raise ValueError("Required tile size must not be equal, exceed original image size or be lower or equal 0")
    times = ceil(log(largest_dimension // required_shape_in_corresponding_max_dim, 2))
    splitted_im = split_arr_4_chunks(image)
    for _ in range(times - 1):
        splitted_im = [split_arr_4_chunks(single_image) for single_image in splitted_im]
        splitted_im = list(itertools.chain.from_iterable(splitted_im))
    return splitted_im


def reconstruct_image_from_4_chunks(list_of_chunks: List[np.ndarray]) -> np.ndarray:
    """
    Function used for merging subimages into one single image.

    :param list_of_chunks: list of 4 images chunks. Left-most should be bottom-left image, second top-left,
    third top-right and fourth bottom-right
    :return: one single image after merging
    """
    left_arr = np.concatenate((list_of_chunks[0], list_of_chunks[1]), axis=1)
    right_arr = np.concatenate((list_of_chunks[3], list_of_chunks[2]), axis=1)
    return np.concatenate((left_arr, right_arr), axis=0)


def reconstruct_image_from_more_chunks(list_of_chunks: List[np.ndarray]) -> np.ndarray:
    """
    Function used for merging subimages into one single image.
    The number of images (length of the list) should be the power of 4

    :param list_of_chunks: list of 4 images chunks. Left-most should be bottom-left image, second top-left,
    third top-right and fourth bottom-right
    :return: one single image after merging
    """
    while len(list_of_chunks) != 4:
        reconstructed = []
        for i in range(0, len(list_of_chunks) - 3, 4):
            reconstructed.append(
                reconstruct_image_from_4_chunks(list_of_chunks[i:(i + 4)]))
        list_of_chunks = reconstructed

    return reconstruct_image_from_4_chunks(list_of_chunks)
