import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib


# Load classes names
data_dir = pathlib.Path('./train/train/')
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])


def load_test_data(folder):
    """
    Load all test images from 'folder'.
    """
    X = []  # Images go here

    # Find all test files from this folder
    files = glob.glob(folder + os.sep + "*.jpg")
    # Load all files
    for name in files:
        # Load image
        img = plt.imread(name)
        X.append(img)

    # Convert python list to contiguous numpy array
    X = np.array(X)

    return X


def generate_submit(test_generator):

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    best_model = tf.keras.models.load_model('best_model.hdf5')
    predictions = best_model.predict_generator(test_generator, steps=nb_samples, verbose=1)
    y_classes = predictions.argmax(axis=-1)
    le = LabelEncoder().fit(CLASS_NAMES)
    labels = list(le.inverse_transform(y_classes))

    print(predictions)
    print(y_classes)
    print(labels)

    new_submission_path = "submission" + ".csv"

    with open(new_submission_path, "w") as fp:
        fp.write("Id,Category\n")
        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))
    print("Submission made!")
