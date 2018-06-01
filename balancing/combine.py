import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import model_selection
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns

import datetime
from sklearn import preprocessing
import numpy as np

from sklearn.ensemble import VotingClassifier

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

print("Loading data file...")
data = pd.read_csv("data_for_student_case.csv", header=0)
data = data.dropna()

data['card_time_since_last_transaction'] = None #17
data['card_total_amount_last_24h'] = None #18
data['card_fraud_ratio'] = None #19
data['card_average_amount'] = None #20
data['card_average_num_of_daily_transactions'] = None #21
data['ip_time_since_last_transaction'] = None #22
data['ip_total_amount_last_24h'] = None #23
data['ip_fraud_ratio'] = None #24
data['ip_average_amount'] = None #25
data['ip_average_num_of_daily_transactions'] = None #26

def time_since_last_transaction(group, transaction):
	if len(group)==0:
		return 0
	return transaction[12]-group[-1][12]
def total_amount_last_24h(group, transaction):
	if len(group)==0:
		return 0
	return sum(map(lambda t: t[5], filter(lambda t: transaction[12]-t[12]<=86400, group)))
def fraud_ratio(group, transaction):
	if len(group)==0:
		return 0
	return sum(map(lambda t: 1 if t[9]=="Chargeback" else 0, group))/len(group)
def average_amount(group, transaction):
	if len(group)==0:
		return 0
	return sum(map(lambda t: t[5], group))/len(group)
	return 0
def average_num_of_daily_transactions(group, transaction):
	if len(group)==0:
		return 0
	#TODO store data in daily format?
	#TODO convert time strings to objects?
	#TODO calculate and return 
	return 0

# print(data.shape)
#print(list(data.columns))

data['creationdate'] = pd.to_datetime(data['creationdate'])
# print(data['creationdate'])
data['creationdate'] = (data['creationdate'] - pd.datetime(1970,1,1)).dt.total_seconds()
# print(data['creationdate'])
data['creationdate'] = data['creationdate'].astype('int64')#//1e9
# print(data['creationdate'])
data.sort_values('creationdate')

ips = {}
cards = {}

#print(data[data.columns.values].values)
for i, transaction in enumerate(data[data.columns.values].values):

	# timestamp = datetime.datetime.strptime(transaction[12], "%Y-%m-%d %H:%M:%S").timestamp()
	# data.iat[i, 12] = timestamp
	# transaction[12] = timestamp

	if transaction[16] not in cards:
		cards[transaction[16]]=[]
	card = cards[transaction[16]]

	data.iat[i,17] = time_since_last_transaction(card, transaction)
	data.iat[i,18] = total_amount_last_24h(card, transaction)
	data.iat[i,19] = fraud_ratio(card, transaction)
	data.iat[i,20] = average_amount(card, transaction)
	data.iat[i,21] = average_num_of_daily_transactions(card, transaction)

	card.append(transaction)

	if transaction[15] not in ips:
		ips[transaction[15]]=[]
	ip = ips[transaction[15]]

	data.iat[i,22] = time_since_last_transaction(ip, transaction)
	data.iat[i,23] = total_amount_last_24h(ip, transaction)
	data.iat[i,24] = fraud_ratio(ip, transaction)
	data.iat[i,25] = average_amount(ip, transaction)
	data.iat[i,26] = average_num_of_daily_transactions(ip, transaction)

	ip.append(transaction)

#print(data[data.columns.values].values)

cards = None
ips = None

data['amount'] = np.log1p(data['amount'].values)

# print(data['simple_journal'].value_counts())
# print(data.groupby('ip_id'))

# '''Group information in columns based on mean of certain features'''
# '''Comment out sections to print out the means for that particular feature'''
# mean_simple_journal = data.groupby('simple_journal').mean()
# #mean_issuercountrycode = data.groupby('issuercountrycode').mean()
# #mean_shopperinteraction = data.groupby('shopperinteraction').mean()
# # print(mean_simple_journal)
# #print(mean_issuercountrycode)

'''Map binary values to each simple_journal value for classification purposes'''
'''Since it is only performance that counts, classifier is more performant with Refused mapped to 1'''
#print(data.shape)
data = data[data['simple_journal'] != 'Refused']
#print(data.shape)
data['simple_journal'] = data['simple_journal'].map({'Chargeback': 1, 'Settled': 0, 'Refused': 1})

# '''BarPlot  for dependent variables in this case simple_journal'''
# '''Comment out print statement to see plot'''
# sns.countplot(x='simple_journal', data=data, palette='hls')
# # plt.show()

# '''BarPlots for other varaiables, comment out print statement for plot'''
# sns.countplot(x='ipverificationcodesupplied', data=data)
# # plt.ylim(0, 25000)
# # plt.show()

# '''Check for missing values if any'''
# '''comment out print statement to see missing values'''
# # print(data.isnull().sum())

'''Drop columns not needed for prediction'''
'''The to be dropped columns are chosen based on count and frequency plots'''
'''['txid', 'bookingdate', 'issuercountrycode', 'txvariantcode', 'bin', 'amount', 
	'currencycode', 'shoppercountrycode', 'shopperinteraction', 'simple_journal', 
	'ipverificationcodesupplied', 'cvcresponsecode', 'creationdate', 'accountcode', 
	'mail_id', 'ip_id', 'ip_id', 'time_since_last_transaction', 'total_amount_last_24h', 
	'fraud_ratio', 'average_amount', 'average_num_of_daily_transactions']'''
#carddata = data.drop(data.columns[[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 21,   22, 23, 24, 25, 26]], axis=1, inplace=False)
#ipdata = data.drop(data.columns[[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]], axis=1, inplace=False)
data.drop(data.columns[[0, 1, 2, 4, 7, 12, 14, 15, 16, 21, 26]], axis=1, inplace=True)

'''Create dummy variables for columns needed for prediction'''
data = pd.get_dummies(data, columns=['txvariantcode', 'currencycode', 'shopperinteraction',
                                       'cardverificationcodesupplied',
                                      'cvcresponsecode', 'accountcode'])

print(data.columns.values)

min_max_scaler = preprocessing.MinMaxScaler()
data[['card_time_since_last_transaction', 'card_total_amount_last_24h', 'card_fraud_ratio', 'card_average_amount', 'ip_time_since_last_transaction', 'ip_total_amount_last_24h', 'ip_fraud_ratio', 'ip_average_amount']] = min_max_scaler.fit_transform(data[['card_time_since_last_transaction', 'card_total_amount_last_24h', 'card_fraud_ratio', 'card_average_amount', 'ip_time_since_last_transaction', 'ip_total_amount_last_24h', 'ip_fraud_ratio', 'ip_average_amount']].values)
# carddata[['time_since_last_transaction', 'total_amount_last_24h', 'fraud_ratio', 'average_amount']] = min_max_scaler.fit_transform(carddata[['time_since_last_transaction', 'total_amount_last_24h', 'fraud_ratio', 'average_amount']].values)
# ipdata[['time_since_last_transaction', 'total_amount_last_24h', 'fraud_ratio', 'average_amount']] = min_max_scaler.fit_transform(ipdata[['time_since_last_transaction', 'total_amount_last_24h', 'fraud_ratio', 'average_amount']].values)

'''Split data into training and test sets'''
'''X contains all rows in data2 and all columns from 1 upwards'''
'''y contains all rows in data2 and column 0'''
#X_data = data.iloc[:,1:]
y_data = data.iloc[:,1]


X_data_train, X_data_test, y_train, y_test = train_test_split(data, y_data, random_state=0)

X_card_train = X_data_train.iloc[:, [0, 2, 3, 4, 5]]
X_card_test = X_data_test.iloc[:, [0, 2, 3, 4, 5]]

X_ip_train = X_data_train.iloc[:, [0, 6, 7, 8, 9]]
X_ip_test = X_data_test.iloc[:, [0, 6, 7, 8, 9]]

X_data_train = X_data_train.iloc[:, 10:]
X_data_test = X_data_test.iloc[:, 10:]

# X_card = carddata.iloc[:, [x for x in range(carddata.shape[1]) if x != 1]]
# y_card = carddata.iloc[:,1]

# X_card_train, X_card_test, y_card_train, y_card_test = train_test_split(X_card, y_card, random_state=0)

# X_ip = ipdata.iloc[:, [x for x in range(ipdata.shape[1]) if x != 1]]
# y_ip = ipdata.iloc[:,1]

# X_ip_train, X_ip_test, y_ip_train, y_ip_test = train_test_split(X_ip, y_ip, random_state=0)


'''Logistic Regression model'''
data_classifier = LogisticRegression(class_weight='balanced', random_state=0)
data_classifier.fit(X_data_train, y_train)

card_classifier = LogisticRegression(class_weight='balanced', random_state=0)
card_classifier.fit(X_card_train, y_train)

ip_classifier = LogisticRegression(class_weight='balanced', random_state=0)
ip_classifier.fit(X_ip_train, y_train)

'''Predict test set results'''
y_data_pred = data_classifier.predict_proba(X_data_test)[:,1]
y_card_pred = card_classifier.predict_proba(X_card_test)[:,1]
y_ip_pred = ip_classifier.predict_proba(X_ip_test)[:,1]

y_pred = list(map(lambda x,y,z: 1 if (x+y+z)/3>0.5 else 0, y_data_pred, y_card_pred, y_ip_pred))
y_pred_proba = list(map(lambda x,y,z: (x+y+z)/3>0.5, y_data_pred, y_card_pred, y_ip_pred))

'''Print Accuracy'''
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

'''compute precision, recall, f-measure and support values'''
print(classification_report(y_test, y_pred))

# '''Plot ROC curve'''

logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()




# '''Visualizations'''
# # pd.crosstab(data.accountcode, data.simple_journal).plot(kind= 'bar')
# # plt.title('Chargedback Frequency for Amount')
# # plt.xlabel('xxx')
# # plt.ylabel('Simple_journal Frequency')
# # plt.ylim(0, 20000)
# # plt.show()
# # plt.savefig('chargeback_freq_per_currencycode')

# '''histograms'''
# # data.bin.hist()
# # plt.title("ip Frequency for bin")
# # plt.xlabel('bin')
# # plt.ylabel('Frequency')
# # plt.show()
# # plt.savefig("Bin histogram")

