from numpy import *
import pandas as pd
from sklearn import preprocessing
from itertools import islice


def sliding_window(arr, n):
	ret = []
	for i,x in enumerate(arr[:-n+1]):
		it = iter(arr[i:])
		ngram = ''.join(map(str, list(islice(it, n))))
		ret.append(ngram)
	return ret

def ngram(states, offset=0):
	ngram = {}
	for i, (state, nstate) in enumerate(zip(states[:-1],states[1:])):
		nx = len(state)+i+offset
		if state in ngram:
			if nstate in ngram[state]:
				(c,x) = ngram[state][nstate]
				x.append(nx)
				ngram[state][nstate] = (c+1,x)
			else:
				ngram[state][nstate] = (1,[nx]) 
		else:
			ngram[state] = {nstate:(1,[nx])}

	return ngram

def combine_ngrams(a,b):
    maxc = 1
    for sfrom in b:
		if sfrom in a:
			for sto in b[sfrom]:
				if sto in a[sfrom]:
					(c,x) = a[sfrom][sto]
					(c2,x2) = b[sfrom][sto]
					a[sfrom][sto] = (c+c2,x+x2)
				else:
					a[sfrom][sto] = b[sfrom][sto]
		else:
			a[sfrom] = b[sfrom]


data = pd.read_csv("BATADAL_dataset03.csv")
test_data = pd.read_csv("BATADAL_test_dataset.csv")

'''Normalise data'''
datanumpy = data[data.columns[1:]].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
datanumpy_scaled = min_max_scaler.fit_transform(datanumpy)
data[data.columns[1:]] = pd.DataFrame(datanumpy_scaled)


datanumpy = test_data[test_data.columns[1:]].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
datanumpy_scaled = min_max_scaler.fit_transform(datanumpy)
test_data[test_data.columns[1:]] = pd.DataFrame(datanumpy_scaled)

'''Plot continuous data'''
#plt.plot(x,data[data.columns[1]])

window_size=6

'''Now for the test data'''
print("Running on test data")
alarms = {}

'''Discretise data into 4 sections'''
for col in ["F_PU1","F_PU6","F_PU7","F_PU11","P_J317","P_J14"]:
	data.loc[data[col] > 0.67, col] = 2
	data.loc[(data[col] > 0.33) & (data[col]<=0.67), col] = 1
	data.loc[data[col] <= 0.33, col] = 0

	data[col] = data[col].astype(int)

	test_data.loc[test_data[col] > 0.67, col] = 2
	test_data.loc[(test_data[col] > 0.33) & (test_data[col]<=0.67), col] = 1
	test_data.loc[test_data[col] <= 0.33, col] = 0

	test_data[col] = test_data[col].astype(int)

	sliding_data = sliding_window(data[col].values, window_size)
	sliding_test_data = sliding_window(test_data[col].values, window_size)

	sliding_test_lbls = test_data[test_data.columns[-1]].values#[window_size:]

	print("Creating n-grams")

	states = ngram(sliding_data)
	states_test = ngram(sliding_test_data, len(sliding_data))

	print(states)
	print(states_test)

	limit = 2
	test_limit = 2

	TP = 0
	FP = 0

	combine_ngrams(states,states_test)

	for sfrom in states:
		for sto in states[sfrom]:
			(c,l) = states[sfrom][sto]
			if c<=test_limit:
				for x in l:
					if x>=len(sliding_data):
						if sliding_test_lbls[x-len(sliding_data)]==1:
							TP+=1
						else:
							FP+=1
						alarms[x-len(sliding_data)] = test_data[test_data.columns[0]].values[x-len(sliding_data)]
					else:
						FP+=1

	print("Sensor %s; TP:%d, FP:%d"%(col,TP,FP))

print("Indices raising alarms:")
print(list(alarms.keys()))
print(list(alarms.values()))