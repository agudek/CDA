import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

from sklearn import preprocessing

df = pd.read_csv("scenario_10_filtered.csv")

#labels
#'Unnamed: 0', 'flows', 'label', 'bytes', 'date', 'time', 'tos', 'flags', 'source_ip', 'source_port', 'destination_ip', 'destination_port', 'protocol', 'packets', 'duration'

'''Plot some columns as subplots under each other'''
#sns.set_color_codes("muted")

columns=['bytes', 'packets', 'duration']

'''Normalise data.'''
datanumpy = df[columns].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
datanumpy_scaled = min_max_scaler.fit_transform(datanumpy)
df[columns] = pd.DataFrame(datanumpy_scaled)

df['label'] = df['label'].map({"LEGITIMATE": 0, "Botnet": 1})

'''Plot some columns as subplots under each other'''

f, axarr = plt.subplots(len(columns), sharex=True)

x = range(len(df[df.columns[0]]))

for i,col in enumerate(columns):
    axarr[i].plot(x, df[col])
    axarr[i].legend([col],loc=2, prop={'size': 6}, handlelength=1)
    # axarr[i].axes.get_yaxis().set_visible(False)
    axarr[i].fill_between(x,df['label'], color=(1,0,0,0.2))

plt.show()
