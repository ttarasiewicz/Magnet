"""Simple IO functions regarding image data."""
import cv2
import numpy as np
from sr_core.image import utils


def save_image(image: np.ndarray, path: str):
    """
    Saves image under given path. Extension must be included in the path string.

    :param image: Image to save.
    :param path: Path for saving the image.
    """
    image = utils.restore_image_dimensions(image)
    if image.ndim == 3:
        if image.dtype in [np.uint8, np.uint16] and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def read_image(path: str) -> np.ndarray:
    """
    Reads image. If color_mode is passed performs additional conversion if necessary.

    :param path: path to file
    :return: numpy.ndarray that is direct representation of an image
    """
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # print(image.shape[2])
    # opencv by default stores RGB value in reverse i.e BGR. Below if converts back to RGB representation.
    if image.ndim == 3:
        if image.dtype in [np.uint8, np.uint16] and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype in [np.uint8, np.uint16] and image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image
