import argparse
from os import makedirs, path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tqdm import tqdm

from common.data_loader import DataLoader
from common.utils import denormalize
from mmtgan.utils import ImageMerger
from mtgan.utils import VALID_CHARS, segment_chars


def normal_eval(input_dir, output_dir, model_path, dis_path=None):
    """
    Evaluate given model.
    :param input_dir: Data input path.
    :param output_dir: Output path to store images.
    :param model_path: Path for the h5 file for the generator network.
    :param dis_path: Optional, path of discriminator
    :return:
    """
    loader = DataLoader(input_dir, 'jpg', test=True)
    generator = load_model(model_path)

    discriminator = None
    rec_output_file = None
    if dis_path:
        discriminator = load_model(dis_path)
        rec_output_file = open(path.join(output_dir, 'recognized.txt'), 'w')

    left = np.array(Image.open('test_img.jpg'))[:, :85]

    for idx, name in tqdm(enumerate(loader.files_names)):
        out = generator.predict(np.array([loader.train_imgs[idx]]))
        img = np.concatenate([left, denormalize(out[0])], 1)
        Image.fromarray(img).save(path.join(output_dir, name))

        if discriminator:
            segments = segment_chars([out[0]])
            recognized = discriminator.predict(segments)
            chars = ''.join(VALID_CHARS[np.argmax(i[:-1])] for i in recognized)
            rec_output_file.write(f'{name} {chars}\n')

    if rec_output_file:
        rec_output_file.close()


def multi_img_eval(input_dir, output_dir, model_path, dis_path, labels_path):
    """
    As `normal_eval` but for models that support multiple images as channels
    """
    multi_imgs = 4
    loader = DataLoader(input_dir, 'jpg', test=True)
    img_merger = ImageMerger(labels_path)
    generator = load_model(model_path)

    discriminator = load_model(dis_path)
    rec_output_file = open(path.join(output_dir, 'recognized.txt'), 'w')

    left = np.array(Image.open('test_img.jpg'))[:, :85]

    for idx, name in tqdm(enumerate(loader.files_names)):
        in_img = img_merger.merge(idx, multi_imgs, loader.train_imgs)
        out = generator.predict(in_img)
        img = np.concatenate([left, denormalize(out[0])], 1)
        Image.fromarray(img).save(path.join(output_dir, name))

        segments = segment_chars([out[0]])
        recognized = discriminator.predict(segments)
        chars = ''.join(VALID_CHARS[np.argmax(i[:-1])] for i in recognized)
        rec_output_file.write(f'{name} {chars}\n')

    if rec_output_file:
        rec_output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir', required=True,
                        dest='input_dir', help='Path for input test images.')

    parser.add_argument('-o', '--output_dir', dest='output_dir',
                        default='./output', required=True,
                        help='Output directory for generated images.')

    parser.add_argument('-m', '--model_path', dest='model_path',
                        help='Path for model h5 data.')

    parser.add_argument('-d', '--discriminator_path',
                        dest='discriminator_path',
                        help=('Path for the discriminator h5 data, to recognize'
                              ' chars. Only on supported GANs'))

    parser.add_argument('-M', '--multi', dest='multi', action='store_true',
                        help='Use multiple images (only for supported models.)')

    parser.add_argument('-l', '--labels', dest='labels',
                        help='Path for the file containing plates\' labels.')
    values = parser.parse_args()

    output_directory = values.output_dir
    makedirs(output_directory, exist_ok=True)

    if values.multi:
        multi_img_eval(
            values.input_dir, output_directory, values.model_path,
            values.discriminator_path, values.labels)
    else:
        normal_eval(
            values.input_dir, output_directory, values.model_path,
            values.discriminator_path)
