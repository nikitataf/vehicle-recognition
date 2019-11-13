"""
TAU Vehicle Type Recognition Competition. Classification
of images of different vehicle types, including cars,
bicycles, vans, ambulances, etc. (total 17 categories).
Main startup script contains most of the initialization.
"""

__author__ = "Nikita Tafintsev"
__copyright__ = "Copyright 2019"
__license__ = "All rights reserved"
__maintainer__ = "Nikita Tafintsev"
__email__ = "nikita@tafintsev.tech"
__status__ = "Beta test"


import argparse
from data import load_data, image_flow
from run import train
from predict import predict
import os
import pathlib
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True' # MacOS specifics -__-


if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser(description='Simulation options')
    parser.add_argument('--train', default='./train/qwe/', type=str, help='Directory of train data')
    parser.add_argument('--test', default='./testtest/', type=str, help='Directory of test data')
    parser.add_argument('--IMG_HEIGHT', default='224', type=int, help='Image height')
    parser.add_argument('--IMG_WIDTH', default='224', type=int, help='Image width')
    parser.add_argument('--weight_decay', default='1e-4', type=float, help='Weight decay')
    parser.add_argument('--batch_size', default='20', type=int, help='Batch size')
    parser.add_argument('--epochs', default='2', type=int, help='Epochs')
    parser.add_argument('--num_classes', default='2', type=int, help='Number of classes')
    args = parser.parse_args()

    # Load pictures into numpy arrays
    # X, y, classes = load_data(args.train)

    # Load classes names
    data_dir = pathlib.Path(args.train)
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

    # Load image generators
    train_generator, validation_generator, test_generator = image_flow(args)

    # Train model
    train(args, train_generator, validation_generator)

    # Predict values
    predict(test_generator, CLASS_NAMES)
