import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

from sklearn.cross_validation import  KFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
'''Read data from csv file using pandas'''
data = pd.read_csv('data_for_student_case.csv')

'''Do not consider Refused transactions in model'''
'''Map chargeback to 1 and settled to 0'''
data = data[data['simple_journal'] != 'Refused']
data['simple_journal'] = data['simple_journal'].map({'Chargeback': 1, 'Settled': 0})

data['card_id'] = [x.strip().replace('card', '') for x in data['card_id']]
data['ip_id'] = [x.strip().replace('ip', '') for x in data['ip_id']]
data['mail_id'] = [x.strip().replace('email', '') for x in data['mail_id']]

data.drop(data.columns[[0, 1, 12]], axis=1, inplace=True)

'''Create dummy variables for columns needed for prediction'''
data = pd.get_dummies(data, columns=['txvariantcode', 'currencycode', 'shopperinteraction', 'issuercountrycode',
                                       'cardverificationcodesupplied', 'shoppercountrycode', 'currencycode',
                                      'cvcresponsecode', 'accountcode'], drop_first=True)


'''Normalize amount feature for transactions'''
data['normalamount'] = StandardScaler().fit_transform(data['amount'].reshape(-1,1))
data = data.drop(['amount'], axis=1)


'''Divide data into test and training set'''
X = data.loc[:, data.columns != 'simple_journal']
y = data.loc[:, data.columns == 'simple_journal']

'''Check number of data points in minority class '''
number_fraud_records = len(data[data.simple_journal ==1])
fraud_indices = np.array(data[data.simple_journal ==1].index)

'''Pick indices of majority class'''
normal_indices = data[data.simple_journal ==0].index

'''Select random indices from the list of indices of the majority class and put them in the form of an array'''
random_normal_indices = np.random.choice(normal_indices, number_fraud_records, replace=False)
random_normal_indices = np.array(random_normal_indices)

'''Append fraud_indices and random_normal indices'''
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

'''Create undersample data from concatenated list of fraud and normal indices'''
'''Create test and training undersampling sets'''
under_sample_data = data.loc[under_sample_indices,:]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'simple_journal']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'simple_journal']

'''Split entire dataset and the undersampled data into training and test sets'''
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)


'''Use 10 kfold to find best model'''
def printing_kfold_scores(x_train_data, y_train_data):
    fold = KFold(len(y_train_data), 10, shuffle=False)
    c_param_range = [0.001,0.01, 0.1, 1, 10,100, 1000, 10000, 100000, 1000000]
    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range
    j=0
    for c_param in c_param_range:
        print('C_parameter:', c_param)
        recall_accs = []
        for iteration, indices in enumerate(fold, start=1):
            lr = LogisticRegression(C=c_param, penalty='l1')
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0],:].values.ravel())
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)

            print('Iteration', iteration, ':recall score=', recall_acc)

        results_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
        j +=1
    best_c = results_table
    best_c.dtypes.eq(object)  # you can see the type of best_c
    new = best_c.columns[best_c.dtypes.eq(object)]  # get the object column of the best_c
    best_c[new] = best_c[new].apply(pd.to_numeric, errors='coerce', axis=0)# change the type of object
    best_c.fillna(0, inplace=True)
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    print('Best model to choose from cross val is with c param =', best_c)
    return best_c

best_c = printing_kfold_scores(X_train_undersample, y_train_undersample)

'''Predict model for undersampled data'''
lr = LogisticRegression(C=best_c, penalty='l1')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(np.nan_to_num(X_test_undersample))

'''create and compute Confusion matrix'''
cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
np.set_printoptions(precision=2)

print(cnf_matrix)

print(classification_report(y_test_undersample, y_pred_undersample))

print("Accuracy of undersampled testing dataset", metrics.accuracy_score(y_test_undersample, y_pred_undersample))
print("recall of undersampled testing dataset: ", metrics.recall_score(y_test_undersample, y_pred_undersample))

'''Plot roc curve'''
lr = LogisticRegression(C = best_c, penalty='l1')
y_pred_undersample_score = lr.fit(X_train_undersample, y_train_undersample.values.ravel()).decision_function(X_test_undersample.values)
fpr, tpr, thresholds = roc_curve(y_test_undersample.values.ravel(), y_pred_undersample_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
