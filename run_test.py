import argparse
from os import makedirs, path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tqdm import tqdm

from common.data_loader import DataLoader
from common.utils import denormalize

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir', action='store',
                        dest='input_dir', default='./data',
                        help='Path for input test images.')

    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir',
                        default='./output',
                        help='Output directory for generated images.')

    parser.add_argument('-m', '--model_path', action='store', dest='model_path',
                        help='Path for model h5 data.')

    left = np.array(Image.open('test_img.jpg'))[:,:85]

    values = parser.parse_args()

    output_dir = values.output_dir
    makedirs(output_dir, exist_ok=True)

    loader = DataLoader(values.input_dir, 'jpg', test=False)
    generator = load_model(values.model_path)

    for idx, name in tqdm(enumerate(loader.files_names)):
        out = generator.predict(np.array([loader.train_imgs[idx]]))
        out = denormalize(out[0])
        out = np.concatenate([left, out], 1)
        Image.fromarray(out).save(path.join(output_dir, name))
