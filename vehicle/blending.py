import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pathlib


# Load classes names
data_dir = pathlib.Path('/Users/tafintse/PycharmProjects/vehicle-recognition/train/train/')
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])


def average_csv(csv_paths):
    if len(csv_paths) < 2:
        print("Blending takes two or more csv files!")
        return

    # Read the first file
    df_blend = pd.read_csv(csv_paths[0], index_col=0)

    # Loop over all files and add them
    for csv_file in csv_paths[1:]:
        df = pd.read_csv(csv_file, index_col=0)
        df_blend = df_blend.add(df)

    # Divide by the number of files
    df_blend = df_blend.div(len(csv_paths))

    # Save the blend file
    # df_blend.to_csv('blend.csv')
    # print(df_blend.head(10))

    predictions = np.array(df_blend)
    y_classes = predictions.argmax(axis=-1)
    le = LabelEncoder().fit(CLASS_NAMES)
    labels = list(le.inverse_transform(y_classes))

    print(predictions)
    print(y_classes)
    print(labels)

    new_submission_path = "blend_submission_avg" + ".csv"

    with open(new_submission_path, "w") as fp:
        fp.write("Id,Category\n")
        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))
    print("Submission made!")


def max_csv(csv_paths):
    if len(csv_paths) < 2:
        print("Blending takes two or more csv files!")
        return

    # Read the first file
    df_blend = pd.read_csv(csv_paths[0], index_col=0)

    # Loop over all files and add them
    for csv_file in csv_paths[1:]:
        df = pd.read_csv(csv_file, index_col=0)
        df_blend = pd.concat([df_blend, df], axis=1, ignore_index=True)

    predictions = np.array(df_blend)
    classes = np.tile(CLASS_NAMES, len(csv_paths))
    y_classes = predictions.argmax(axis=-1)
    le = LabelEncoder().fit_transform(classes)

    y_classes_new = []
    for i in y_classes:
        value = le[i]
        y_classes_new.append(value)

    y_classes_new = np.array(y_classes_new)
    le = LabelEncoder().fit(classes)

    labels = list(le.inverse_transform(y_classes_new))

    print(predictions)
    print(y_classes)
    print(labels)

    new_submission_path = "blend_submission_max" + ".csv"

    with open(new_submission_path, "w") as fp:
        fp.write("Id,Category\n")
        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))
    print("Submission made!")


def weighted_average_csv(csv_paths, weights):
    if len(csv_paths) < 2:
        print("Blending takes two or more csv files!")
        return

    # Read the first file
    df_blend = pd.read_csv(csv_paths[0], index_col=0)
    df_blend = df_blend.mul(weights.loc[0,:], axis=1)

    # Loop over all files and add them
    for i in range(1, len(csv_paths)):
        csv_file = csv_paths[i]
        df = pd.read_csv(csv_file, index_col=0)
        df = df.mul(weights.loc[i, :], axis=1)
        df_blend = df_blend.add(df)

    # Divide by the number of files
    df_blend = df_blend.div(len(csv_paths))

    # Save the blend file
    # df_blend.to_csv('blend.csv')
    # print(df_blend.head(10))

    predictions = np.array(df_blend)
    y_classes = predictions.argmax(axis=-1)
    le = LabelEncoder().fit(CLASS_NAMES)
    labels = list(le.inverse_transform(y_classes))

    print(predictions)
    print(y_classes)
    print(labels)

    new_submission_path = "blend_submission_weight_avg" + ".csv"

    with open(new_submission_path, "w") as fp:
        fp.write("Id,Category\n")
        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))
    print("Submission made!")


if __name__ == "__main__":

    # Load csv names
    data_dir = pathlib.Path('./probabilities/')
    list_probs = list(['./probabilities/' + item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])

    # average_csv(list_probs)
    # max_csv(list_probs)

    # Weights for individual category
    weights = pd.DataFrame({'Ambulance': [1, 1, 1],
                            'Barge': [1, 1, 1],
                            'Bicycle': [1, 1, 1],
                            'Boat': [1, 1, 1],
                            'Bus': [1, 1, 1],
                            'Car': [10, 1, 1],
                            'Cart': [1, 1, 1],
                            'Caterpillar': [1, 1, 1],
                            'Helicopter': [1, 1, 1],
                            'Limousine': [1, 1, 1],
                            'Motorcycle': [1, 1, 1],
                            'Segway': [1, 1, 1],
                            'Snowmobile': [1, 1, 1],
                            'Tank': [1, 1, 1],
                            'Taxi': [1, 1, 1],
                            'Truck': [1, 1, 1],
                            'Van': [1, 1, 1],
                            })

    weighted_average_csv(list_probs, weights)
