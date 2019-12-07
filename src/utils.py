import numpy as np

def normalize(batch_img: np.ndarray):
    """Transform image (in batch) from [0, 255] to [-1, 1]."""
    batch_img = batch_img.astype('float32')
    return batch_img / 127.5 - 1

def denormalize(batch_img: np.ndarray):
    """Transform image (in batch) from [-1, 1] to [0, 255]."""
    return (batch_img + 1) * 127.5


