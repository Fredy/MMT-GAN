from collections import defaultdict
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from common.utils import denormalize


class ImageMerger:
    """Merge multiple images in the channel axis."""

    def __init__(self, labels_path):
        self.related_imgs = self._get_plates_related(labels_path)

    def _get_plates_related(self, labels_path):
        """
        Return list of list, first index is the index of the image, the second
        list is the indexes of the different images that have the same plate.
        """
        labels_tmp = list()
        labels_relation = defaultdict(list)
        with open(labels_path) as labels_file:
            for idx, line in enumerate(labels_file):
                label = line.split(' ')[1].strip()
                labels_tmp.append(label)
                labels_relation[label].append(idx)

        return [labels_relation[label] for label in labels_tmp]

    def merge(self, indexes, n_images, image_array):
        """
        Return images merged in channel, e.g: if the images have shape (10,20,3)
        then the returned array will be of shape (10,20, 3 * `n_images`)

        :param indexes: Indexes of the images that will be considered.
        :param n_images: Number of merged images
        :param image_array: Array of all the images
        :return: Array with merged images in the last axis.
        """
        if not isinstance(indexes, Iterable):
            indexes = [indexes]

        imgs = list()
        for i in indexes:
            random_ints = np.random.choice(self.related_imgs[i], n_images)
            img = image_array[random_ints]
            img = np.concatenate(img, 2)
            imgs.append(img)

        return np.array(imgs)


def plot_example_images(
        output_dir, epoch, generator, label_imgs, test_imgs, imgs_count,
        img_merger, merge_imgs_count, examples=4):
    """
    Create a plot with original (LR) images, generated images and label (HR)
    images.

    :param output_dir:
    :param epoch:
    :param generator:
    :param label_imgs:
    :param test_imgs:
    :param imgs_count:
    :param examples:
    :return:
    """
    random_ints = np.random.randint(0, imgs_count, examples)
    image_batch_hr = denormalize(label_imgs[random_ints])
    image_batch_lr = test_imgs[random_ints]
    lr_merged = img_merger.merge(random_ints, merge_imgs_count, test_imgs)
    gen_img = generator.predict(lr_merged)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)

    plt.clf()

    for idx in range(examples):
        plt.subplot(examples, 3, 1 + idx * 3)
        plt.imshow(image_batch_lr[idx], interpolation='nearest')
        plt.axis('off')

        plt.subplot(examples, 3, 2 + idx * 3)
        plt.imshow(generated_image[idx], interpolation='nearest')
        plt.axis('off')

        plt.subplot(examples, 3, 3 + idx * 3)
        plt.imshow(image_batch_hr[idx], interpolation='nearest')
        plt.axis('off')

    plt.subplots_adjust(wspace=0.01)
    plt.savefig(
        path.join(output_dir, f'generated_image_{epoch}.svg'),
        bbox_inches='tight', pad_inches=0, )
