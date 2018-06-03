from numpy import *
import numpy as np
import pandas as pd
from sklearn import preprocessing
from statsmodels.multivariate.pca import PCA

data = pd.read_csv("BATADAL_dataset03.csv")
test_data = pd.read_csv("BATADAL_test_dataset.csv")
#test_data = pd.read_csv("BATADAL_dataset04.csv")

'''Normalise data'''
# datanumpy = data[data.columns[1:]].values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# datanumpy_scaled = min_max_scaler.fit_transform(datanumpy)
# data[data.columns[1:]] = pd.DataFrame(datanumpy_scaled)


# datanumpy = test_data[test_data.columns[1:]].values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# datanumpy_scaled = min_max_scaler.fit_transform(datanumpy)
# test_data[test_data.columns[1:]] = pd.DataFrame(datanumpy_scaled)

pca_train = PCA(data[data.columns[1:-1]], method='eig', normalize=True, tol_em=5e-06)

fig = pca_train.plot_rsquare()
fig.show()


print(pca_train.factors)