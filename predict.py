import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import glob
import numpy as np
import os
import matplotlib.pyplot as plt


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


def predict(test_generator, class_names):

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    best_model = tf.keras.models.load_model('best_model.hdf5')
    predictions = best_model.predict_generator(test_generator, steps=nb_samples)
    y_classes = predictions.argmax(axis=-1)
    le = LabelEncoder().fit(class_names)
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


if __name__ == "__main__":

    print('Predictions')
