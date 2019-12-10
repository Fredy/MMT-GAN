import numpy as np
import tensorflow as tf

from eval.data_loader import DataLoader


def _img_comparator(dir_a, dir_b, output_path, func):
    loader = DataLoader('jpg', dir_a, dir_b)
    res = list()
    with open(output_path, 'w') as out_file:
        for idx, name in enumerate(loader.files_names):
            diff = func(
                tf.convert_to_tensor(loader.imgs[0][idx]),
                tf.convert_to_tensor(loader.imgs[1][idx]), 255)
            diff = diff.numpy()
            out_file.write(f'{name} {diff}\n')
            res.append(diff)

    return np.array(res)


def psnr(dir_a, dir_b, output_path):
    return _img_comparator(dir_a, dir_b, output_path, tf.image.psnr)


def ssim(dir_a, dir_b, output_path):
    return _img_comparator(dir_a, dir_b, output_path, tf.image.ssim)


def _count_fails_chars(recognized, label):
    count = 0
    for r, l in zip(recognized, label):
        if r != l:
            count += 1

    return count


def accuracy(path_gen, path_label, output_path):
    """
    Compare recognized chars from `path_gen` file to labels in `path_label`.
    Save the number of differences between two strings in `output_path`.
    """

    with open(path_gen) as recognized, open(path_label) as label, open(
            output_path, 'w') as output:
        for line_rec, line_label in zip(recognized, label):
            name, label = line_label.split(' ')
            chars_label = label.strip()
            chars_recognized = line_rec.split(' ')[1].strip()

            count_fails = _count_fails_chars(chars_recognized, chars_label)
            output.write(f'{name}\t{count_fails}\n')
