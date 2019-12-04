import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pathlib


# Load classes names
data_dir = pathlib.Path('C:/Users/Roman/Dropbox/TUT/wolfram/Competition/train')
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

    new_submission_path = "blend_submission_avg" + ".csv"

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

    # weights = pd.DataFrame({'Ambulance': [1,1,1,1],
    #                         'Barge': [1,1,1,1],
    #                         'Bicycle': [1,1,1,1],
    #                         'Boat': [1,1,1,1],
    #                         'Bus': [1,1,1,1],
    #                         'Car': [1,1,1,1],
    #                         'Cart': [1,1,1,1],
    #                         'Caterpillar': [1,1,1,1],
    #                         'Helicopter': [1,1,1,1],
    #                         'Limousine': [1,1,1,1],
    #                         'Motorcycle': [1,1,1,1],
    #                         'Segway': [1,1,1,1],
    #                         'Snowmobile': [1,1,1,1],
    #                         'Tank': [1,1,1,1],
    #                         'Taxi': [1,1,1,1],
    #                         'Truck': [1,1,1,1],
    #                         'Van': [1,1,1,1],
    #                         })

    # weights = pd.DataFrame({'Ambulance': [0.7692307692307693,0.7379679144385025,0.819672131147541,0.8020833333333334],
    #                         'Barge': [0.07407407407407407,0.07142857142857142,0.,0.06060606060606061],
    #                         'Bicycle': [0.8981288981288982,0.9055793991416309,0.9155370177267987,0.9112426035502958],
    #                         'Boat': [0.9501488095238095,0.9452662721893491,0.9407870540639941,0.9448223733938019],
    #                         'Bus': [0.9246861924686192,0.9319148936170213,0.9224318658280922,0.9399141630901288],
    #                         'Car': [0.9319130559953936,0.9270697404668696,0.9285507665606015,0.9243174423539398],
    #                         'Cart': [0.6885245901639344,0.75,0.7513227513227513,0.6436781609195402],
    #                         'Caterpillar': [0.9829059829059829,0.9741379310344828,0.9655172413793104,0.9743589743589743],
    #                         'Helicopter': [0.979746835443038,0.9733840304182508,0.9704749679075737,0.9787234042553191],
    #                         'Limousine': [0.71875,0.7142857142857143,0.6329113924050633,0.6493506493506493],
    #                         'Motorcycle': [0.8977900552486188,0.8697916666666666,0.897680763983629,0.9068702290076336],
    #                         'Segway': [0.9343065693430657,0.9007633587786259,0.9465648854961832,0.9552238805970149],
    #                         'Snowmobile': [0.9459459459459459,0.9577464788732394,0.9210526315789473,0.8372093023255814],
    #                         'Tank': [0.9130434782608695,0.8995633187772926,0.8977777777777778,0.9043478260869565],
    #                         'Taxi': [0.7096774193548387,0.7425149700598802,0.7368421052631579,0.7571428571428571],
    #                         'Truck': [0.7234957020057307,0.7042857142857143,0.7172512526843235,0.7090252707581227],
    #                         'Van': [0.7198697068403909,0.7211093990755008,0.7060702875399361,0.7147766323024055],
    #                         })

    weights = pd.DataFrame({'Ambulance': [0.7954545454545454,0.7419354838709677,0.8426966292134831,0.7857142857142856],
                            'Barge': [0.14285714285714285,0.125,0.,0.07692307692307693],
                            'Bicycle': [0.9133192389006343,0.9525959367945822,0.9340425531914893,0.88],
                            'Boat': [0.9487369985141159,0.9383259911894273,0.9288307915758897,0.9585889570552147],
                            'Bus': [0.9020408163265307,0.9240506329113923,0.9016393442622951,0.9399141630901288],
                            'Car': [0.9404416037187682,0.9425117924528302,0.9416251100029334,0.9166666666666666],
                            'Cart': [0.875,0.8888888888888888,0.9102564102564102,0.8888888888888888],
                            'Caterpillar': [0.9913793103448276,0.9912280701754386,0.9824561403508771,0.9827586206896551],
                            'Helicopter': [0.9723618090452262,0.9672544080604534,0.9767441860465116,0.9606879606879607],
                            'Limousine': [0.7931034482758621,0.7142857142857143,0.5681818181818182,0.5952380952380952],
                            'Motorcycle': [0.8485639686684073,0.7822014051522248,0.8392857142857142,0.945859872611465],
                            'Segway': [0.9014084507042254,0.9076923076923077,0.9538461538461539,0.9411764705882353],
                            'Snowmobile': [0.9210526315789473,0.9714285714285714,0.875,0.72],
                            'Tank': [0.9051724137931034,0.8956521739130435,0.9099099099099099,0.896551724137931],
                            'Taxi': [0.5945945945945946,0.6739130434782609,0.65625,0.8153846153846154],
                            'Truck': [0.7122708039492243,0.6914446002805049,0.7056338028169014,0.7034383954154728],
                            'Van': [0.7038216560509554,0.670487106017192,0.6779141104294478,0.7375886524822695],
                            })

    # weights = pd.DataFrame({'Ambulance': [0.7446808510638298,0.7340425531914894,0.7978723404255319,0.8191489361702128],
    #                         'Barge': [0.05,0.05,0.,0.05],
    #                         'Bicycle': [0.8834355828220859,0.8629856850715747,0.8977505112474438,0.9447852760736196],
    #                         'Boat': [0.9515648286140089,0.9523099850968704,0.9530551415797317,0.9314456035767511],
    #                         'Bus': [0.9484978540772532,0.9399141630901288,0.944206008583691,0.9399141630901288],
    #                         'Car': [0.9235378031383737,0.9121255349500713,0.9158345221112696,0.9320970042796005],
    #                         'Cart': [0.5675675675675675,0.6486486486486487,0.6396396396396397,0.5045045045045045],
    #                         'Caterpillar': [0.9745762711864406,0.9576271186440678,0.9491525423728814,0.9661016949152542],
    #                         'Helicopter': [0.9872448979591836,0.9795918367346937,0.9642857142857142,0.9974489795918365],
    #                         'Limousine': [0.6571428571428571,0.7142857142857143,0.7142857142857143,0.7142857142857143],
    #                         'Motorcycle': [0.9530791788856305,0.9794721407624634,0.9648093841642229,0.8709677419354839],
    #                         'Segway': [0.9696969696969697,0.8939393939393939,0.9393939393939394,0.9696969696969697],
    #                         'Snowmobile': [0.9722222222222222,0.9444444444444444,0.9722222222222222,1.],
    #                         'Tank': [0.9210526315789473,0.9035087719298246,0.8859649122807017,0.9122807017543859],
    #                         'Taxi': [0.88,0.8266666666666667,0.84,0.7066666666666667],
    #                         'Truck': [0.735080058224163,0.7176128093158661,0.7292576419213974,0.7147016011644832],
    #                         'Van': [0.7366666666666667,0.78,0.7366666666666667,0.6933333333333334],
    #                         })

    weighted_average_csv(list_probs, weights)
