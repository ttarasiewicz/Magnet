"""
Module with transformations that can be applied to images.

"""
import enum
import cv2
import numpy as np

from typing import Tuple, Union, Iterable
from scipy import ndimage
from sr_core.image import utils, resolution

Filter = resolution.Filter


class MakeShapeDivisibleBy:
    class Axis(enum.IntEnum):
        X = 0
        Y = 1

    def __init__(self, factor: int):
        super().__init__()
        self._factor = factor

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Crops borders of an image so it is divisible by given factor.

        :param image: Image to be cropped.
        :return: Cropped image.
        """
        remove_rows = image.shape[0] % self._factor
        remove_cols = image.shape[1] % self._factor
        cropped = np.array(image)
        if remove_rows != 0:
            cropped = cropped[:-remove_rows, :]
        if remove_cols != 0:
            cropped = cropped[:, :-remove_cols]
        return cropped


class MoveAxis:
    def __init__(self, target: Union[int, Iterable[int]], destination: Union[int, Iterable[int]]):
        super().__init__()
        self._target = target
        self._destination = destination

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Move axes of an array to new positions.
        Other axes remain in their original order.

        :param image: Target image.
        :return: Target image with one axis moved to destined position.
        """
        result = np.moveaxis(image, self._target, self._destination)
        return result


class Normalize:
    def __init__(self):
        super().__init__()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        mean = np.mean(image)
        std = np.std(image)
        if std < 1e-5:
            std = 1.
        return (image - mean) / std


class ToFloat:
    def __init__(self):
        super(ToFloat, self).__init__()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32)


class ToDouble:
    def __init__(self):
        super(ToDouble, self).__init__()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float64)


class Multiply:
    def __init__(self, value: float):
        super(Multiply, self).__init__()
        self._value = value

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image * self._value


class Divide:
    def __init__(self, value: float):
        super(Divide, self).__init__()
        self._value = value

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image / self._value


class Flip:
    class Axis(enum.IntEnum):
        X = 0
        Y = 1

    def __init__(self, axis: Axis):
        super().__init__()
        self._axis = axis

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Flips image in given axis.

        :param image: Image to be flipped.
        :return: Flipped image.
        """
        return np.flip(image, self._axis)


class Resize:
    def __init__(self, size: Tuple[int, int], interpolation: Filter):
        super().__init__()
        self._size = size
        self._interpolation = interpolation

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes an image to given size.

        :param image: Image to be resized.
        :return: Resized image.
        """
        return resolution.resize_image(image, self._size[1], self._size[0], self._interpolation)


class Scale:
    def __init__(self, scale_factor: float, interpolation: Filter):
        super().__init__()
        self._scale_factor = scale_factor
        self._interpolation = interpolation

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Scales image by given scale_factor .

        :param image: Image to be scaled.
        :return: Scaled image.
        """
        return resolution.scale_image_resolution(image, self._scale_factor, self._interpolation)


class ToGrayscale:
    class From(enum.IntEnum):
        RGB = cv2.COLOR_RGB2GRAY
        RGBA = cv2.COLOR_RGBA2GRAY
        BGR = cv2.COLOR_BGR2GRAY

    def __init__(self, from_colorspace: From):
        super().__init__()
        self._from_colorspace = from_colorspace

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Changes image colorspace to grayscale.

        :param image: Image to be grayscaled.
        :return: Grayed image.
        """
        return cv2.cvtColor(image, self._from_colorspace)


class AveragingBlur:
    """
    Averaging blur as described in https://www.tutorialspoint.com/opencv/opencv_blur_averaging.htm.

    """

    def __init__(self, kernel_size: Tuple[int, int]):
        self._kernel_size = kernel_size
        super().__init__()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies averaging blur to an image.

        :param image: Image to be blurred.
        :return: Blurred image.
        """
        blurred = cv2.blur(image, self._kernel_size)
        return blurred


class GaussianBlur:
    """
    Gaussian blur as described in https://www.tutorialspoint.com/opencv/opencv_gaussian_blur.htm

    """

    def __init__(self, kernel_size: Tuple[int, int], sigma_x: float, sigma_y: float):
        """
        For thorough description of parameters visit
        https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1

        :param kernel_size: Size of the filter.
        :param sigma_x: Standard deviation in X direction.
        :param sigma_y: Standard deviation in Y direction.
        """
        self._kernel_size = kernel_size
        self._sigma_x = sigma_x
        self._sigma_y = sigma_y
        super().__init__()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies gaussian blur to an image.

        :param image: Image to be blurred.
        :return: Blurred image.
        """
        blurred = cv2.GaussianBlur(image, self._kernel_size, sigmaX=self._sigma_x, sigmaY=self._sigma_y)
        return blurred


class MedianBlur:
    """
    Median blur as described in https://www.tutorialspoint.com/opencv/opencv_median_blur.htm.

    """

    def __init__(self, kernel_size: int):
        self._kernel_size = kernel_size
        super().__init__()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies median blur to an image.

        :param image: Image to be blurred.
        :return: Blurred image.
        """
        blurred = cv2.medianBlur(image, self._kernel_size)
        return blurred


class SaltAndPepperNoise:
    """
    Adds salt and pepper noise to an image.

    """

    def __init__(self, saturation: float):
        """

        :param saturation: Value in percents so it should be from 0 to 1. The bigger it its the more noise would
        be added. Values from 0.001 up to 0.01 are generally good candidates.
        :raises ValueError: When saturation is not in range from 0.0 to 1.0.
        """
        if saturation > 1.0 or saturation < 0.0:
            raise ValueError(f"saturation must be between 1.0 and 0.0. It is {saturation}.")
        self._saturation = saturation

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies noise.

        :param image: Image to which noise will be applied.
        :return: Noised image.
        """
        minimum, maximum = utils.get_values_range_for_dtype(image.dtype)
        saturation = self._saturation * maximum
        noise_map = np.random.uniform(minimum, maximum, image.shape[:2])
        image = np.array(image)
        image[noise_map > (maximum - saturation)] = maximum
        image[noise_map < (minimum + saturation)] = minimum

        return image


class PoissonNoise:
    """
    Applies poisson (shot) noise to an image.

    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies noise.

        :param image: Image to which noise will be applied.
        :return: Noised image.
        """
        return np.random.poisson(image)


class JPEGNoise:
    """
    Applies JPEG noise to an image. Quality values of 60 and below are giving visible degradation, while higher values
    are less noticeable.

    """

    def __init__(self, quality: int = 95):
        """

        :param quality: Quality where 100 means no degradation and 0 means maximum degradation.
        :raises ValueError: When quality is not between 0 and 100.
        """
        if quality > 100 or quality < 0:
            raise ValueError(f"Quality must be set to value between 0 and 100, not {quality}.")
        self._quality = quality

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies noise.

        :param image: Image to which noise will be applied.
        :return: Noised image.
        """
        _, buffer = cv2.imencode(".jpg", image, params=[cv2.IMWRITE_JPEG_QUALITY, self._quality])
        return cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)


class Rotation:
    """
    Rotate image by a given angle.

    """

    def __init__(self, degrees: float, border_mode: int = cv2.BORDER_DEFAULT):
        self._degrees = degrees
        self._border_mode = border_mode

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Rotates image.

        :param image: Image to rotate.
        :return: Rotated image.
        """
        m = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), self._degrees, 1)
        return cv2.warpAffine(image, m, image.shape[1::-1], borderMode=self._border_mode)


class SubPixelShift:
    """
    Perform sub-pixel shift.
    First image is shifted by a given amount of pixels.
    Then the lost edges are being cropped. At last image is being down-scaled.
    By doing so we achieve the artificial sub-pixel shift.

    """

    class Shift:
        def __init__(self, x_direction: int, y_direction: int):
            self.x_direction = x_direction
            self.y_direction = y_direction

    def __init__(self, shift: Shift, scale: float, interpolation: resolution.Filter = resolution.Filter.LANCZOS):
        self._shift = shift
        self._scale = scale
        self._interpolation = interpolation

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Shifts an image.

        :param image: Image to shift.
        :return: Shifted image.
        """
        shift = [self._shift.y_direction, self._shift.x_direction]
        if image.ndim == 3:
            shift.append(0)  # Do not perform any shift on 3 dimension if exists

        shifted_image = ndimage.shift(image, shift)
        cropped_image = shifted_image
        if shift[0] >= 0:
            cropped_image = cropped_image[shift[0]:, :]
        else:
            cropped_image = cropped_image[:shift[0], :]
        if shift[1] >= 0:
            cropped_image = cropped_image[:, shift[1]:]
        else:
            cropped_image = cropped_image[:, :shift[1]]
        scaled_image = resolution.scale_image_resolution(cropped_image, self._scale,
                                                         scaling_method=self._interpolation)

        return scaled_image
