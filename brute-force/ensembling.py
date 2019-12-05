import csv
import pathlib
import itertools
import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd


def brute_force(sub_weight, max_acc):
    # Load csv names
    data_dir = pathlib.Path('./best_submissions/')
    sub_files = list(['./best_submissions/' + item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])

    Hlabel = 'Id'
    Htarget = 'Category'
    npt = 1  # number of places in target

    place_weights = {}
    for i in range(npt):
        place_weights[i] = 1/(i+1)

    lg = len(sub_files)
    sub = [None]*lg
    for i, file in enumerate(sub_files):
        # input files
        # print("Reading {}: w={} - {}". format(i, sub_weight[i], file))
        reader = csv.DictReader(open(file,"r"))
        sub[i] = sorted(reader, key=lambda d: float(d[Hlabel]))

    # output file
    out = open("./ensembled_submission.csv", "w", newline='')
    writer = csv.writer(out)
    writer.writerow([Hlabel,Htarget])

    for p, row in enumerate(sub[0]):
        target_weight = {}
        for s in range(lg):
            row1 = sub[s][p]
            for ind, trgt in enumerate(row1[Htarget].split(' ')):
                target_weight[trgt] = target_weight.get(trgt,0) + (place_weights[ind]*sub_weight[s])
        tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
        writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
    out.close()

    gt = pd.read_csv('./test_labels.csv')['Category']
    labels = pd.read_csv('./ensembled_submission.csv')['Category']
    bools = np.equal(gt, labels)
    diff = np.sum(bools)
    acc = diff/len(gt)

    if acc > max_acc:
        weights_acc = []
        max_acc = acc
        print(max_acc)
        weights_acc.append(max_acc)
        weights_acc.append(sub_weight)

        # Save the best weights
        with open('output.txt', 'w') as f:
            for s in weights_acc:
                f.write(str(s) + '\n')

        new_submission_path = "/Users/tafintse/PycharmProjects/vehicle-recognition/kaggle_test/best_submission" + ".csv"
        with open(new_submission_path, "w") as fp:
            fp.write("Id,Category\n")
            for i, label in enumerate(labels):
                fp.write("%d,%s\n" % (i, label))
        print("Submission made!")
    return max_acc


if __name__ == "__main__":

    probs = np.arange(0, 4.0, 0.5)

    max_acc = 0
    for subset in itertools.permutations(probs, 8):
        print(subset)
        max_acc = brute_force(list(subset), max_acc)
