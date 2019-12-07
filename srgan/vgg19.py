import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model


class VGGLoss(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
        vgg19 = VGG19(
            include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(
            inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False

        return K.mean(K.square(model(y_true) - model(y_pred)))


