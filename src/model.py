from tensorflow.keras.layers import (Input, Conv2D, ReLU, BatchNormalization,
                                     Dense, Add, Conv2DTranspose, LeakyReLU,
                                     Flatten)
from tensorflow.keras.models import Model


def get_generator(input_shape):
    in_layer = Input(input_shape)
    net = Conv2D(filters=64, kernel_size=9, strides=1, padding='same',
                 activation='relu')(in_layer)

    # ======= N residual blocks (N = 16)========
    for i in range(16):
        # filters, kernel_size, strides, padding
        block = Conv2D(64, 3, 1, 'same')(net)
        block = BatchNormalization()(block)
        block = ReLU()(block)
        block = Conv2D(64, 3, 1, 'same')(block)
        block = BatchNormalization()(block)
        block = Add()([net, block])
        net = block
    # ======= N residual blocks ========
    # stride = 1, to maintain the image size
    net = Conv2DTranspose(64, 3, 1, 'same', activation='relu')(net)
    net = Conv2DTranspose(3, 1, 1, 'same', activation='tanh')(net)

    generator = Model(inputs=in_layer, outputs=net, name='Generator')

    return generator


def _discriminator_block(model, filters, kernel_size, strides):
    tmp = Conv2D(filters, kernel_size, strides, 'same')(model)
    tmp = BatchNormalization()(tmp)
    tmp = LeakyReLU(alpha=0.2)(tmp)

    return tmp


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

    net = Dense(35, activation='sigmoid')(net)

    discriminator = Model(inputs=in_layer, outputs=net)

    return discriminator
