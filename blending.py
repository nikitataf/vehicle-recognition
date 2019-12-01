import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pathlib


# Load classes names
data_dir = pathlib.Path('./train/train/')
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])


def blend_csv(csv_paths):
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
    df_blend.to_csv('blend.csv')
    print(df_blend.head(10))

    predictions = np.array(df_blend)
    y_classes = predictions.argmax(axis=-1)
    le = LabelEncoder().fit(CLASS_NAMES)
    labels = list(le.inverse_transform(y_classes))

    print(predictions)
    print(y_classes)
    print(labels)

    new_submission_path = "blend_submission" + ".csv"

    with open(new_submission_path, "w") as fp:
        fp.write("Id,Category\n")
        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))
    print("Submission made!")


# Replace this with two or more CSV files
blend_csv(['./models_prob/Inception-BatchNorm_accuracy=0.883262_probs.csv',
           './models_prob/ResNet-50_accuracy=0.882131_probs.csv',
           './models_prob/ResNet-101_accuracy=0.884519_probs.csv',
           './models_prob/ResNet-152_accuracy=0.885021_probs.csv',
           './models_prob/ResNeXt-50_accuracy=0.89809_probs.csv',
           './models_prob/ResNeXt-101_accuracy=0.889545_probs.csv'
           ])
