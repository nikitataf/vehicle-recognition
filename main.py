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
from data import load_data
from run import train
from predict import predict


if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser(description='Simulation options')
    parser.add_argument('--train', default='train/scaled_train', type=str, help='Directory of train data')
    parser.add_argument('--test', default='test/scaled_test', type=str, help='Directory of test data')
    parser.add_argument('--weight_decay', default='1e-4', type=float, help='Weight decay')
    parser.add_argument('--batch_size', default='128', type=int, help='Batch size')
    parser.add_argument('--eps', default='10', type=int, help='Epochs')
    parser.add_argument('--num_classes', default='17', type=int, help='Number of classes')
    args = parser.parse_args()

    # Load pictures into numpy arrays
    X, y, classes = load_data(args.train)
    print("X shape: " + str(X.shape))
    print("y shape: " + str(y.shape))

    # Train model
    train(X, y, args)

    # Predict values
    predict(args.test, classes)
