import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_ubyte, img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(folder):
    """
    Load all images from subdirectories of
    'folder'. The subdirectory name indicates
    the class.
    """

    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here

    subdirectories = glob.glob(folder + "/*")

    # Loop over all folders
    for d in subdirectories:

        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")

        # Load all files
        for name in files:

            # Load image and parse class name
            img = imread(name)
            class_name = name.split(os.sep)[-2]

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)

            class_idx = classes.index(class_name)

            X.append(img)
            y.append(class_idx)

    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    classes = np.array(classes)

    return X, y, classes


def resize_test_data():
    root = './test/testset/'
    new_path = './test/scaled_test/'
    os.mkdir(new_path)
    i = 0
    for file in sorted(os.listdir(root)):
        img = imread(root + file)
        res = resize(img, (200, 200))
        imsave(new_path + os.sep + file, img_as_float(res))
        i = i + 1
        print(str(i) + ' images out of ' + str(len(os.listdir(root))) + ' processed')

    print('Successfully resized')


def resize_train_data():
    root = './train/train/'
    new_path = './train/scaled_train/'
    os.mkdir(new_path)
    i = 0
    # Rescale all files in each subdirectory
    for category in sorted(os.listdir(root)):
        os.mkdir(new_path + category)
        for file in sorted(os.listdir(os.path.join(root, category))):
            img = imread(root + category + os.sep + file)
            res = resize(img, (200, 200))
            imsave(new_path + os.sep + category + os.sep + file, img_as_float(res))
            i = i + 1
            print(str(i) + ' images processed')

    print('Successfully resized')


def image_flow(args):

    # Data Generator for the train data
    data_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.33)
    # Generator for the test data
    test_generator = ImageDataGenerator(rescale=1. / 255)

    train_generator = data_generator.flow_from_directory(args.train, target_size=(args.IMG_HEIGHT, args.IMG_WIDTH),
                                                         shuffle=True,
                                                         seed=13,
                                                         class_mode='categorical', batch_size=args.batch_size,
                                                         subset="training")

    validation_generator = data_generator.flow_from_directory(args.train, target_size=(args.IMG_HEIGHT, args.IMG_WIDTH),
                                                              shuffle=True, seed=13,
                                                              class_mode='categorical', batch_size=args.batch_size,
                                                              subset="validation")

    test_generator = test_generator.flow_from_directory(args.test, target_size=(args.IMG_HEIGHT, args.IMG_WIDTH),
                                                        shuffle=False,
                                                        batch_size=1)

    return train_generator, validation_generator, test_generator


if __name__ == "__main__":

    # resize_test_data()
    resize_train_data()
