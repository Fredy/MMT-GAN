import numpy as np
import tensorflow as tf

VALID_CHARS = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
CHARS_IDX = {c: i for i, c in enumerate(VALID_CHARS)}

CHARS = 4  # Number of chars in the plate
H_START = 8
H_END = 44 + H_START
W_START = 0
CHAR_WIDTH = 23


def chars_to_one_hot(chars, dtype=np.float32):
    """
    Return one hot vector of chars + real/fake label:
        10 digits + 24 letters (excluding I, O) + label
    Label value is set to 0.

    :param chars: String of valid chars
    :param dtype: Dtype of the output array, defaults to np.float32
    :return: Array(chars len, 35)
    """
    nchars = len(chars)
    out = np.zeros((nchars, len(VALID_CHARS) + 1), dtype)
    for i, c in enumerate(chars):
        out[i, CHARS_IDX[c]] = 1

    return out


def segment_chars(batch):
    """
    Segment chars of a 60x180 plate
    :param batch: Array like with shape (B, H, W, 3)
        B: batch size
        W: image width
        H: image height
    :return: Array like with shape (B*N, 44,44,3)
        N: number of chars
    """
    chars = list()
    for img in batch:
        for j in range(CHARS):
            w_start = W_START + (j * CHAR_WIDTH)
            w_end = W_START + ((j + 1) * CHAR_WIDTH)
            block = img[H_START:H_END, w_start:w_end]
            chars.append(block)

    return np.array(chars)


def get_one_hot_labels(labels_path):
    """Return one hot vector for all the labels in `labels_path`."""

    one_hot = list()
    with open(labels_path) as labels_file:
        for line in labels_file:
            label = line.split(' ')[1].strip()
            one_hot.append(chars_to_one_hot(label))

    return np.array(one_hot)

class SegmentationLayer(tf.keras.layers.Layer):

  def call(self, input, **kwargs):
    return input



def segment_chars_tensor(batch, position):
    """
    Segment chars of a 60x180 plate
    :param batch: Array like with shape (B, H, W, 3)
        B: batch size
        W: image width
        H: image height
    :return: Array like with shape (B*N, 44,44,3)
        N: number of chars
    """
    w_start = W_START + (position * CHAR_WIDTH)
    w_end = W_START + ((position + 1) * CHAR_WIDTH)
    return batch[:, H_START:H_END, w_start:w_end]
