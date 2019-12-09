from collections.abc import Iterable
from functools import lru_cache
from os import path

import numpy as np
from PIL import Image

from common.utils import get_files_names_from_dir, normalize

TRAIN_DIR = 'train'
TEST_DIR = 'test'
LABEL_DIR = 'label'


class DataGetter:
    def __init__(self, get_img_func):
        self.get_img_func = get_img_func

    def __getitem__(self, item):
        if not isinstance(item, Iterable):
            return self.get_img_func(item)

        return np.array([self.get_img_func(i) for i in item])


class DataLoader:
    """Load training data without consuming much RAM."""

    def __init__(self, directory, ext, test=False):
        """
        The specified directory should contains two folders: `train` and `label`
        each one with the same quantity of images. Each image in the `train`
        folder should have it pair (with the same name) in the `label` folder.

        :param directory: Directory containing `train` and `label` folders.
        :param ext: Image extension.
        :return: Train and label images.
        """
        if not test:
            self.train_dir = path.join(directory, TRAIN_DIR)
        else:
            self.train_dir = path.join(directory, TEST_DIR)
        self.label_dir = path.join(directory, LABEL_DIR)
        self.files_names = get_files_names_from_dir(self.train_dir, ext)

        self.label_imgs = DataGetter(self._get_label_img)
        self.train_imgs = DataGetter(self._get_train_img)

    @lru_cache(2048)
    def _get_train_img(self, idx):
        full_path = path.join(self.train_dir, self.files_names[idx])
        return normalize(np.array(Image.open(full_path))[:, 85:, :])

    @lru_cache(2048)
    def _get_label_img(self, idx):
        full_path = path.join(self.label_dir, self.files_names[idx])
        return normalize(np.array(Image.open(full_path))[:, 85:, :])

    @property
    def imgs_count(self):
        return len(self.files_names)
