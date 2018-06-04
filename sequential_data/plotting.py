import pandas as pd
from pandas import datetime
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import plotly.offline as py
import plotly.figure_factory as ff


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings("ignore")

def parser(x):
	return datetime.strptime(x, '%d/%m/%y %H')

data = pd.read_csv("BATADAL_dataset03.csv", header=0, parse_dates=[0], index_col=None, squeeze=True, date_parser=parser)

'''Get correlations for dataset'''
correlation = data.corr()

'''Convert Datetime column values from String to datetime'''
data["DATETIME"] = pd.to_datetime(data["DATETIME"])
data.index = data["DATETIME"]
del data["DATETIME"]

lt1 = pd.DataFrame(data[['L_T1']])

'''consider just the measurements for these 2 days so as to reduce the Lags for ARMA model'''
reducedLags = lt1['2014-04-06':'2014-04-07']
print(reducedLags.count())

'''Resample data on Hourly basis'''
reducedLagsValues = reducedLags.resample("H").mean()

'''Plot partial autocorrelation Graph for reduced Lags'''
plot_pacf(reducedLagsValues)

'''Plot autocorrelation function for reduced lags'''
plot_acf(reducedLagsValues)


'''Plot Correlation'''
correlation = np.array(correlation)
for i in range(len(correlation)):
    for j in range(len(correlation)):
        correlation[i, j] = round(correlation[i, j], 2)

columns = ['L_T1', 'L_T2' ,'L_T3' ,'L_T4' ,'L_T5' ,'L_T6' ,'L_T7' ,'F_PU1',
 'S_PU1' ,'F_PU2' ,'S_PU2' ,'F_PU3', 'S_PU3', 'F_PU4', 'S_PU4' , 'F_PU5', 'S_PU5',
 'F_PU6' ,'S_PU6' ,'F_PU7' ,'S_PU7' ,'F_PU8' ,'S_PU8' ,'F_PU9' ,'S_PU9', 'F_PU10',
 'S_PU10' ,'F_PU11', 'S_PU11' ,'F_V2' ,'S_V2', 'P_J280' ,'P_J269', 'P_J300',
 'P_J256' ,'P_J289', 'P_J415' ,'P_J302' ,'P_J306' ,'P_J307', 'P_J317', 'P_J14',
 'P_J422' ,'ATT_FLAG']
minimum = 0
maximum = 40
fig = ff.create_annotated_heatmap(correlation[minimum:maximum, minimum:maximum], x=columns[minimum:maximum], y=columns[minimum:maximum])
fig.layout.title = 'Heatmap of correlation between variables'

py.plot(fig)
plt.show()


