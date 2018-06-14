from numpy import *
import pandas as pd

from sklearn import preprocessing
from itertools import islice

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data1 = pd.read_csv("BATADAL_dataset03.csv")
data1 = data1.drop(['DATETIME','ATT_FLAG'], axis=1)
data1numpy = data1[data1.columns].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
data1numpy_scaled = min_max_scaler.fit_transform(data1numpy)
data1[data1.columns] = pd.DataFrame(data1numpy_scaled)

def sliding_window(arr, n):
	ret = []
	for i,x in enumerate(arr[:-n]):
		it = iter(arr[i:])
		ret.append(list(islice(it, n)))
	return ret


print("SENSOR, MAE, MSE")

for col in data1.columns:
	values = data1[col].values
	window_size=100
	sliding_window_data=sliding_window(values,window_size)
	sliding_window_next=values[range(window_size,len(values))]

	names=range(window_size)
	df = pd.DataFrame(sliding_window_data,columns=names)


	cf = DecisionTreeRegressor(max_depth=10)

	X = pd.DataFrame.as_matrix(df)
	y =sliding_window_next

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.01, random_state=0)

	cf.fit(X_train, y_train)

	pred = cf.predict(X_test)

	error_sum = 0
	squared_error_sum = 0
	for i,p in enumerate(pred):
		print("actual %f predicted %f"%(y_test[i],p))
		error_sum=error_sum+(abs(y_test[i]-p))
		squared_error_sum=squared_error_sum+(pow(y_test[i]-p,2))

	n=len(pred)
	print("%s, %f, %f"%(col, error_sum/n, squared_error_sum/n))

