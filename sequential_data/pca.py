from numpy import *
import numpy as np
import pandas as pd
from pandas import  datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def parser(x):
    return datetime.strptime(x, '%d/%m/%y %H')

parsed_train_data1 = pd.read_csv("BATADAL_dataset03.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
parsed_train_data2 = pd.read_csv("BATADAL_dataset04.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
parsed_test_data = pd.read_csv("BATADAL_test_dataset.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

def set_attacks(test_set):
    test_set = test_set.assign(ATT_FLAG=np.zeros(test_set.shape[:][0]))
    test_set.loc[:,'ATT_FLAG'] = test_set.loc[:, 'ATT_FLAG'].map({0: -999})
    test_set.loc['2017-01-16 09:00:00':'2017-01-19 06:00:00', 'ATT_FLAG'] = 1
    test_set.loc['2017-01-30 08:00:00':'2017-02-02 00:00:00', 'ATT_FLAG'] = 1
    test_set.loc['2017-02-09 03:00:00':'2017-02-10 09:00:00', 'ATT_FLAG'] = 1
    test_set.loc['2017-02-12 01:00:00':'2017-02-13 07:00:00', 'ATT_FLAG'] = 1
    test_set.loc['2017-02-24 05:00:00':'2017-02-28 08:00:00', 'ATT_FLAG'] = 1
    test_set.loc['2017-03-10 14:00:00':'2017-03-13 21:00:00', 'ATT_FLAG'] = 1
    test_set.loc['2017-03-25 20:00:00':'2017-03-27 01:00:00', 'ATT_FLAG'] = 1

    return test_set

def transform_data(dataset):
    scalar = StandardScaler()
    x = dataset.iloc[:,0:43].values
    x = scalar.fit_transform(x)
    y = dataset.iloc[:, 43]

    return x,y

def pca_task(pca, t, index, label):
    pca_model = pca.transform(t)
    residual = t - pca.inverse_transform(pca_model)
    spe = np.square(np.linalg.norm(residual, axis = 1))
    spe = spe / max(spe)

    threshold = 0.09
    notanomalous = spe > threshold

    plt.plot(spe)
    plt.plot(label.map({-999: -0.1, 0: 0, 1: 1}).values)
    plt.show()

    plt.figure(figsize=[15, 10])
    plt.plot(notanomalous)
    plt.plot(label.map({-999: -0.1, 0: 0, 1: 1}).values)
    plt.show()

    label = label.values
    predict = notanomalous

    print("*********PCA EValuation Results***********")
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(predict)):
        if (predict[i] == 1):
            if (label[i] == 1):
                TP = TP + 1
            else:
                FP = FP + 1
        else:
            if (label[i] == 1):
                FN = FN + 1
            else:
                TN = TN + 1

    Precision = float(TP) / float(TP + FP)
    Recall = float(TP) / float(TP + FN)

    F1 = 2 * float(Precision * Recall) / float(Precision + Recall)

    numberOfAnomalousRegions = 0
    detected = 0
    for i in range(len(predict)):
        if (label[i] == 1):
            if (detected == 1):
                continue
            else:
                if (predict[i] == 1):
                    detected = 1
                    numberOfAnomalousRegions = numberOfAnomalousRegions + 1
                continue
        else:
            if (detected == 1):
                detected = 0
            continue
    print('Total true positives ' +str(TP))
    print('Total false negative ' + str(FN))

    print('Precision= ' + str(Precision))
    print('Recall= ' + str(Recall))
    print('F1= ' + str(F1))
    print('number of anomalous regions= ' + str(numberOfAnomalousRegions))


test_set = set_attacks(parsed_test_data)

test_set = test_set.fillna(test_set.median(axis=0))
training_set1 = parsed_train_data1.fillna(parsed_train_data1.median(axis = 0))
training_set2 = parsed_train_data2.fillna(parsed_train_data2.median(axis = 0))

x1, y1 = transform_data(training_set1)
x2, y2 = transform_data(training_set2)
z, yz = transform_data(test_set)

pca = PCA(n_components=7)
pca.fit(x1)

pca_task(pca, x2, training_set1.index, y2)


