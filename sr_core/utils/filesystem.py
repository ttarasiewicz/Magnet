import filecmp
import os
import typing
from typing import Iterator, Union


class DirectoryContents:
    _IMAGE_EXTENSIONS = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.tiff', '.tif')

    @staticmethod
    def list_subdirs(directory_name: str, allowed_dirs: Union[str, list] = None) -> typing.List[os.DirEntry]:
        """
        Returns a list of subdirectries (returned as os.DirEntry) in a folder "directory_name".
        "allowed_dirs" specifies what folders should be added to the list (useful for choosing
        specific Sentinel-2 bands) - if empty, all folders are added.
        """
        if allowed_dirs is None:
            allowed_dirs = []
        elif isinstance(allowed_dirs, str):
            allowed_dirs = [allowed_dirs]
        entries = [entry for entry in os.scandir(directory_name)]
        return [entry for entry in entries if entry.is_dir() and (entry.name in allowed_dirs or allowed_dirs == [])]

    @staticmethod
    def deep_list_subdirs(directory_name: str or os.DirEntry, deepness: int = 0,
                          allowed_dirs: list = None) -> typing.List[os.DirEntry]:
        """
        Returns a list of subdirectories (returned as os.DirEntry) for all folders located
        specific number of levels deeper ("deepness" argument - if 0, works as list_subdirs )
        in a folder "directory_name".
        "allowed_dirs" specifies what folders should be added to the list (useful for choosing
        specific Sentinel-2 bands) - if empty, all folders are added.
        """
        assert deepness >= 0, f"Deepness has to be equal or greater than 0 ({deepness} given)"
        final_list = []
        if deepness == 0:
            final_list += DirectoryContents.list_subdirs(directory_name, allowed_dirs=allowed_dirs)
        else:
            for subdir in DirectoryContents.list_subdirs(directory_name):
                final_list += DirectoryContents.deep_list_subdirs(subdir,
                                                                  deepness=deepness - 1,
                                                                  allowed_dirs=allowed_dirs)

        return final_list

    @staticmethod
    def list_images_only(directory_name: str) -> typing.List[os.DirEntry]:
        """
        Returns a list of images (returned as os.DirEntry) in a folder "directory_name".
        @see DirectoryContents._IMAGE_EXTENSIONS for files treated as images.
        """
        file_entries = [entry for entry in os.scandir(directory_name) if entry.is_file()]
        return [file_entry for file_entry in file_entries if DirectoryContents._is_image_file(file_entry.name)]

    @staticmethod
    def _is_image_file(file_name: str):
        return file_name.endswith(DirectoryContents._IMAGE_EXTENSIONS)


def get_depth(path: str, depth: int = 0) -> int:
    """
    Function scans depth of the given directory.

    :param path: Path to the directory
    :param depth: Depth at which function is at the moment
    :return: Depth of the directory
    """
    if not os.path.isdir(path):
        return depth
    maxdepth = depth
    for entry in os.listdir(path):
        fullpath = os.path.join(path, entry)
        maxdepth = max(maxdepth, get_depth(fullpath, depth + 1))
    return maxdepth


def scantree(path: str) -> Iterator[os.DirEntry]:
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry


def has_common_structure(dir1: str, dir2: str) -> bool:
    """
    Checks if two dirs share structure and content

    :return: Bool indicating whether two directories have common structure.
    """
    for current_ref, current_target in zip(os.walk(dir1), os.walk(dir2)):
        cmp = filecmp.dircmp(current_ref[0], current_target[0])
        if len(cmp.left_only) != 0 and len(cmp.right_only) != 0:
            return False
    return True