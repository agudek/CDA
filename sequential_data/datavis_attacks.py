# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import *
import numpy as np

import pandas as pd

from sklearn import preprocessing

data = pd.read_csv("BATADAL_dataset04.csv")

'''Normalise data. This will also convert attack flag into 0 (no attack) and 1 (attack).'''
datanumpy = data[data.columns[1:]].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
datanumpy_scaled = min_max_scaler.fit_transform(datanumpy)
data[data.columns[1:]] = pd.DataFrame(datanumpy_scaled)

'''Plot some columns as subplots under each other'''
columns=data.columns[1:-1]

f, axarr = plt.subplots(len(columns), sharex=True)

x = range(len(data[data.columns[0]]))

for i,col in enumerate(columns):
    axarr[i].plot(x, data[col])
    axarr[i].legend([col],loc=2, prop={'size': 6}, handlelength=1)
    #axarr[i].axes.get_xaxis().set_visible(False)
    axarr[i].axes.get_yaxis().set_visible(False)
    axarr[i].fill_between(x,data[data.columns[-1]], color=(1,0,0,0.2))

plt.show()
