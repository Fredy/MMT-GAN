import argparse
import os

from srgan.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir',
                        default='./data',
                        help='Traning images directory path.')

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

    train(
        values.epochs, values.batch_size, values.input_dir, output_dir,
        checkpoint_dir)