import argparse
import os

import tensorflow as tf

from mtgan.train import train as mtgan_train
from srgan.train import train as srgan_train

tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir',
                        default='./data',
                        help='Training images directory path.')

    parser.add_argument('-l', '--plate_labels', action='store',
                        dest='plate_labels',
                        help='File container license plates labels')

    parser.add_argument('-n', '--network', action='store', dest='network',
                        help='Network to train.', choices=['srgan', 'mtgan'])

    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir',
                        default='./output',
                        help='Train examples output directory path.')

    parser.add_argument('-c', '--checkpoint_dir', action='store',
                        dest='checkpoint_dir', default='./checkpoints',
                        help='Path to save model checkpoints.')

    parser.add_argument('-b', '--batch_size', action='store', dest='batch_size',
                        default=64,
                        help='Images batch size.', type=int)

    parser.add_argument('-e', '--epochs', action='store', dest='epochs',
                        default=1000,
                        help='Epoch for training the network.', type=int)

    values = parser.parse_args()

    output_dir = values.output_dir
    checkpoint_dir = values.checkpoint_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if values.network == 'srgan':
        srgan_train(
            values.epochs, values.batch_size, values.input_dir, output_dir,
            checkpoint_dir)
    elif values.network == 'mtgan':
        mtgan_train(
            values.epochs, values.batch_size, values.input_dir,
            values.plate_labels, output_dir, checkpoint_dir)
