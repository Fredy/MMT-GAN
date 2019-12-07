from tensorflow.keras.layers import (
    Add, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, Input,
    LeakyReLU, PReLU,
)
from tensorflow.keras.models import Model


def get_generator(input_shape):
    in_layer = Input(input_shape)
    net = Conv2D(
        filters=64, kernel_size=9, strides=1, padding='same')(in_layer)
    net = PReLU(shared_axes=[1, 2])(net)

    pre_res_block = net

    # ======= N residual blocks (N = 16)========
    for _ in range(16):
        block = Conv2D(64, 3, 1, 'same')(net)
        block = BatchNormalization()(block)
        block = PReLU(shared_axes=[1, 2])(block)
        block = Conv2D(64, 3, 1, 'same')(block)
        block = BatchNormalization(momentum=0.5)(block)
        net = Add()([net, block])
    # ======= N residual blocks ========

    # Post residual blocks
    net = Conv2D(64, 3, 1, 'same')(net)
    net = BatchNormalization()(net)
    net = Add()([net, pre_res_block])

    # Upscale
    # TODO: add activation = relu ?
    net = Conv2DTranspose(256, 3, 1, 'same')(net)
    net = Conv2DTranspose(256, 3, 1, 'same')(net)

    net = Conv2D(3, 9, 1, 'same', activation='tanh')(net)

    return Model(inputs=in_layer, outputs=net, name='Generator')


def get_discriminator(input_shape):
    in_layer = Input(input_shape)

    net = Conv2D(64, 3, 1, 'same')(in_layer)
    net = LeakyReLU(alpha=0.2)(net)

    net = _discriminator_block(net, 64, 3, 2)
    net = _discriminator_block(net, 128, 3, 1)
    net = _discriminator_block(net, 128, 3, 2)
    net = _discriminator_block(net, 256, 3, 1)
    net = _discriminator_block(net, 256, 3, 2)
    net = _discriminator_block(net, 512, 3, 1)
    net = _discriminator_block(net, 512, 3, 2)

    net = Flatten()(net)
    net = Dense(1024)(net)
    net = LeakyReLU(alpha=0.2)(net)

    net = Dense(1, activation='sigmoid')(net)

    return Model(inputs=in_layer, outputs=net)


def _discriminator_block(model, filters, kernel_size, strides):
    tmp = Conv2D(filters, kernel_size, strides, 'same')(model)
    tmp = BatchNormalization()(tmp)
    tmp = LeakyReLU(alpha=0.2)(tmp)

    return tmp
