import logging
import re
from os import path

import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.framework.ops import disable_eager_execution
from tqdm import tqdm

from common import utils
from common.data_loader import DataLoader
from common.utils import get_files_names_from_dir
from srgan.model import get_discriminator, get_generator

RE_DISCRIMINATOR_H5 = re.compile(r'dis_model_(\d+)\.h5')

np.random.seed(123)
image_shape = (60, 95, 3)

logging.basicConfig(level=logging.INFO)


def get_last_saved_epoch(model_save_dir):
    """
    Return last save file epoch number.
    :param model_save_dir:
    :return:
    """
    files_names = get_files_names_from_dir(model_save_dir, 'h5')
    if not files_names:
        return 0

    max_save = 0
    for name in files_names:
        match = RE_DISCRIMINATOR_H5.match(name)
        if match:
            new_save = int(match.group(1))
            max_save = max(max_save, new_save)

    return max_save


def get_optimizer():
    return Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


# Combined network
def get_gan_network(discriminator, generator, shape, optimizer, vgg_loss=None):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=['mean_squared_error', 'binary_crossentropy'],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


def train(epochs, batch_size, input_dir, output_dir, model_save_dir):
    disable_eager_execution()
    img_loader = DataLoader(input_dir, 'jpg')
    train_imgs = img_loader.train_imgs
    label_imgs = img_loader.label_imgs

    image_count = img_loader.imgs_count

    batch_count = image_count // batch_size


    saved_epoch = get_last_saved_epoch(model_save_dir)

    if saved_epoch:
        first_epoch = saved_epoch + 1

        generator = load_model(
            path.join(model_save_dir, f'gen_model_{saved_epoch}.h5'))

        discriminator = load_model(
            path.join(model_save_dir, f'dis_model_{saved_epoch}.h5'))
        optimizer = generator.optimizer
    else:
        first_epoch = 1

        generator = get_generator(image_shape)
        discriminator = get_discriminator(image_shape)
        optimizer = get_optimizer()
        generator.compile(loss='mean_squared_error', optimizer=optimizer)
        discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

    gan = get_gan_network(
        discriminator, generator, image_shape, optimizer)

    loss_file = open(path.join(model_save_dir, 'losses.txt'), 'w+')
    loss_file.close()

    for e in range(first_epoch, epochs + 1):
        logging.info(f'======== Epoch {e} =======')
        for _ in tqdm(range(batch_count)):
            rand_ints = np.random.randint(0, image_count, batch_size)

            image_batch_train = train_imgs[rand_ints]
            image_batch_label = label_imgs[rand_ints]
            generated_images_sr = generator.predict(image_batch_train)

            # TODO shouldn't this be ones and zeroes ????
            fake_data_target = np.random.random_sample(batch_size) * 0.2
            real_data_target = np.ones(batch_size) - fake_data_target

            discriminator.trainable = True

            d_loss_real = discriminator.train_on_batch(
                image_batch_label, real_data_target)
            d_loss_fake = discriminator.train_on_batch(
                generated_images_sr, fake_data_target)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            rand_ints = np.random.randint(0, image_count, batch_size)
            image_batch_train = train_imgs[rand_ints]
            image_batch_label = label_imgs[rand_ints]

            gan_target = np.ones(batch_size) - np.random.random_sample(
                batch_size) * 0.2
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(image_batch_train,
                                          [image_batch_label, gan_target])

        logging.info(f'discriminator_loss : {discriminator_loss}')
        logging.info(f'gan_loss : {gan_loss}')
        gan_loss = str(gan_loss)

        with open(path.join(model_save_dir, 'losses.txt'), 'a') as loss_file:
            loss_file.write(
                f'epoch {e} : gan_loss = {gan_loss} ; '
                f'discriminator_loss = {discriminator_loss}\n'
            )

        if e == 1 or e % 5 == 0:
            utils.plot_example_images(
                output_dir, e, generator, label_imgs, train_imgs,
                img_loader.imgs_count)
        if e % 10 == 0:
            generator.save(path.join(model_save_dir, f'gen_model_{e}.h5'))
            discriminator.save(path.join(model_save_dir, f'dis_model_{e}.h5'))
