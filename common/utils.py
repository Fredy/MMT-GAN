from os import path

import matplotlib.pyplot as plt
import numpy as np


def normalize(batch_img: np.ndarray) -> np.ndarray:
    """Transform image (in batch) from [0, 255] to [-1, 1]."""
    batch_img = batch_img.astype('float32')
    return batch_img / 127.5 - 1


def denormalize(batch_img: np.ndarray) -> np.ndarray:
    """Transform image (in batch) from [-1, 1] to [0, 255]."""
    return np.uint8((batch_img + 1) * 127.5)


def plot_example_images(
        output_dir, epoch, generator, label_imgs, test_imgs, imgs_count,
        examples=4):
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
    gen_img = generator.predict(image_batch_lr)
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
