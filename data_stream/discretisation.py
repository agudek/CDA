import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

df = pd.read_csv('scenario_10_filtered.csv')
cols = ['bytes','packets']
df['label'] = df['label'].map({"LEGITIMATE": 0, "Botnet": 1})

'''Define split values (obtained by observing the data)'''
cutoffs = [(0,0.05),(0.05,0.1),(0.1,0.2),(0.2,0.4),(0.4,1.001)]
cutoffs.reverse()

'''Normalise data'''
datanumpy = df[cols].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
datanumpy_scaled = min_max_scaler.fit_transform(datanumpy)
df[cols] = pd.DataFrame(datanumpy_scaled)

print(cutoffs)

'''Discretise data into 5 sections'''
for col in cols:

	for i,(lower, upper) in enumerate(cutoffs):
		df.loc[(df[col] >= lower) & (df[col]<upper), col] = 4-i

	df[col] = df[col].astype(int)


'''Plot some columns as subplots under each other'''

f, axarr = plt.subplots(len(cols), sharex=True)

x = range(len(df[df.columns[0]]))

for i,col in enumerate(cols):
    axarr[i].plot(x, df[col])
    axarr[i].legend([col],loc=2, prop={'size': 6}, handlelength=1)
    # axarr[i].axes.get_yaxis().set_visible(False)
    axarr[i].fill_between(x,df['label'], color=(1,0,0,0.2))

plt.show()

'''Create new discretised dataset and write to csv file'''
df['combined_discretisation'] = df['bytes'].astype(str)+df['packets'].astype(str)
df.to_csv('scenario_10_discretised.csv')