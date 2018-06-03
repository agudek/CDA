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

data1 = pd.read_csv("BATADAL_dataset03.csv")
#data2 = pd.read_csv("BATADAL_dataset04.csv")
#data3 = pd.read_csv("BATADAL_test_dataset.csv")

'''Plot some columns as subplots under each other'''
#sns.set_color_codes("muted")

columns=data1.columns[1:-1]

#f, axarr = plt.subplots(len(data1.columns)-1, sharex=True)
f, axarr = plt.subplots(len(columns), sharex=True)

x = range(len(data1[data1.columns[0]]))




for i,col in enumerate(columns):
    axarr[i].plot(x, data1[col])
    axarr[i].legend([col],loc=2, prop={'size': 6}, handlelength=1)
    axarr[i].axes.get_xaxis().set_visible(False)
    axarr[i].axes.get_yaxis().set_visible(False)

'''Plot multiple columns in single graph after normalisation'''
columns=['L_T3','S_PU4','F_PU4','S_PU5','F_PU5']
plt.figure()
data1numpy = data1[columns].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
data1numpy_scaled = min_max_scaler.fit_transform(data1numpy)
data1[columns] = pd.DataFrame(data1numpy_scaled)

plt.plot(x,data1[columns])
plt.legend(columns)

plt.show()