from os import path

import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from common import utils
from common.data_loader import DataLoader
from srgan.model import get_discriminator, get_generator
from srgan.vgg19 import VGGLoss

np.random.seed(123)
image_shape = (60, 180, 3)


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
    img_loader = DataLoader(input_dir, 'jpg')
    train_imgs = img_loader.train_imgs
    label_imgs = img_loader.label_imgs

    image_count = img_loader.imgs_count

    batch_count = image_count // batch_size

    generator = get_generator(image_shape)
    discriminator = get_discriminator(image_shape)

    optimizer = get_optimizer()
    generator.compile(loss='mean_squared_error', optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

    gan = get_gan_network(
        discriminator, generator, image_shape, optimizer)

    loss_file = open(path.join(model_save_dir, 'losses.txt'), 'w+')
    loss_file.close()

    for e in range(1, epochs + 1):
        print('-' * 15, f'Epoch {e}', '-' * 15)
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

        print(f'discriminator_loss : {discriminator_loss:.5}')
        print(f'gan_loss : {gan_loss:.5}')
        gan_loss = str(gan_loss)

        with open(model_save_dir + 'losses.txt', 'a') as loss_file:
            loss_file.write(
                f'epoch {e} : gan_loss = {gan_loss:.6} ; '
                f'discriminator_loss = {discriminator_loss:.6}\n'
            )

        if e == 1 or e % 5 == 0:
            utils.plot_example_images(
                output_dir, e, generator, label_imgs, train_imgs,
                img_loader.imgs_count)
        if e % 500 == 0:
            generator.save(model_save_dir + f'gen_model_{e}.h5')
            discriminator.save(model_save_dir + f'dis_model_{e}.h5')
