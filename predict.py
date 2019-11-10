from keras.models import Sequential, load_model, Model
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


def predict(folder, classes):
    # Load pictures into numpy array
    X = load_test_data(folder)
    print("X shape: " + str(X.shape))
    print("Classes shape: " + str(classes.shape))

    best_model = load_model('best_model.hdf5')
    predictions = best_model.predict_classes(X)
    le = LabelEncoder().fit(classes)
    labels = list(le.inverse_transform(predictions))

    new_submission_path = "submission" + ".csv"

    with open(new_submission_path, "w") as fp:
        fp.write("Id,Category\n")
        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))
    print("Submission made!")


if __name__ == "__main__":
    print('Predictions')
