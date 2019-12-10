from collections.abc import Iterable
from functools import lru_cache
from os import path

import numpy as np
from PIL import Image

from common.utils import get_files_names_from_dir, normalize


class DataGetter:
    def __init__(self, get_img_func):
        self.get_img_func = get_img_func

    def __getitem__(self, item):
        if not isinstance(item, Iterable):
            return self.get_img_func(item)

        return np.array([self.get_img_func(i) for i in item])


class DataLoader:
    """Load training data without consuming much RAM."""

    def __init__(self, ext, *directories):
        """
        The specified directory should contains two folders: `train` and `label`
        each one with the same quantity of images. Each image in the `train`
        folder should have it pair (with the same name) in the `label` folder.

        :param directory: Directory containing `train` and `label` folders.
        :param ext: Image extension.
        :return: Train and label images.
        """
        self.dirs = directories
        self.files_names = get_files_names_from_dir(self.dirs[0], ext)

        # TODO: fix this
        self.imgs = [
            DataGetter(lambda x: self._get_img(self.dirs[0], x)),
            DataGetter(lambda x: self._get_img(self.dirs[1], x))
        ]

    def _get_img(self, directory, idx, norm=False):
        full_path = path.join(directory, self.files_names[idx])
        if norm:
            return normalize(np.array(Image.open(full_path))[:, 85:, :])
        else:
            return np.array(Image.open(full_path))[:, 85:, :]

    @property
    def imgs_count(self):
        return len(self.files_names)
