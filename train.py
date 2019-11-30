"""
TAU Vehicle Type Recognition Competition. Classification
of images of different vehicle types, including cars,
bicycles, vans, ambulances, etc. (total 17 categories).
Train script contains most of the initialization and training.
"""

__author__ = "Nikita Tafintsev"
__copyright__ = "Copyright 2019"
__license__ = "All rights reserved"
__maintainer__ = "Nikita Tafintsev"
__email__ = "nikita@tafintsev.tech"
__status__ = "Beta test"


import argparse
from dataset import VehicleDataset
from model import train
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # MacOS issue


def main():
    # Parameters
    parser = argparse.ArgumentParser(description='Vehicle Recognition Competition')
    parser.add_argument('--num_classes', default='17', type=int, help='Number of classes')
    parser.add_argument('--train', default='./train/train/', type=str, help='Directory of train data')
    parser.add_argument('--test', default='./test/', type=str, help='Directory of test data')
    parser.add_argument('--IMG_HEIGHT', default='224', type=int, help='Image height')
    parser.add_argument('--IMG_WIDTH', default='224', type=int, help='Image width')
    parser.add_argument('--batch_size', default='10', type=int, help='Batch size')
    parser.add_argument('--epochs', default='100', type=int, help='Epochs')
    parser.add_argument('--weight_decay', default='1e-4', type=float, help='Weight decay')

    args = parser.parse_args()

    # Load image generators
    train_dataset = VehicleDataset(args)

    # Train model
    train(args, train_dataset.train_generator, train_dataset.validation_generator)

    # TEMPORARY HERE
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from postprocessing import generate_submit
    # Generator for the test data
    test_generator = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_generator.flow_from_directory(args.test, target_size=(args.IMG_HEIGHT, args.IMG_WIDTH),
                                                        shuffle=False, batch_size=1)

    # Predict values
    generate_submit(test_generator)


if __name__ == "__main__":
    main()
