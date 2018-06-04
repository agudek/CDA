import pandas as pd

from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings("ignore")

def parser(x):
	return datetime.strptime(x, '%d/%m/%y %H')

'''load data'''
data = pd.read_csv("BATADAL_dataset03.csv", header=0, parse_dates=[0], index_col=None, squeeze=True, date_parser=parser)

'''The 'MS' string groups the data in buckets by start of the month'''
data = data.set_index('DATETIME').resample('D').mean()

'''The term bfill means that we use the value before filling in missing values'''
data = data.fillna(data.bfill())
main_df = pd.DataFrame()

for c in data.columns[[0, 1, 2, 3, 4, 5, 6]]:
    # print(data[c])
    order_selection = sm.tsa.arma_order_select_ic(data[c].values, max_ar = 4, max_ma = 2, ic = "aic")
    ticker = [c]

    df_aic_min = pd.DataFrame([order_selection["aic_min_order"]], index=
    ticker)

    main_df = main_df.append(df_aic_min)
    main_df.to_csv("aic_min_orders.csv")

'''Try different p and q values  in order to get the best aic, bic and hqic values'''
arma_column15 = sm.tsa.ARMA(data['L_T1'].values, (4,0)).fit()
print ("column1Arma5 {}".format(arma_column15.params))
#
arma_column17 = sm.tsa.ARMA(data['L_T1'].values, (3,2)).fit()
print ("column1Arma7 {}".format(arma_column17.params))

arma_column19 = sm.tsa.ARMA(data['L_T1'].values, (3,0)).fit()
print ("column1Arma9 {}".format(arma_column17.params))

arma_column21 = sm.tsa.ARMA(data['L_T1'].values, (2,2)).fit()
print ("column1Arma9 {}".format(arma_column21.params))

print (arma_column15.aic, arma_column15.bic, arma_column15.hqic)
print (arma_column17.aic, arma_column17.bic, arma_column17.hqic)
print (arma_column19.aic, arma_column19.bic, arma_column19.hqic)
print (arma_column21.aic, arma_column21.bic, arma_column21.hqic)

'''divide dataset into test and training data'''
X = data['L_T1'].values
size = int(len(X) * 0.7)
print(size)
train, test = X[0: size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
'''Fit the ARMA model with the p and q values with lowest aic value'''
for t in range(len(test)):
    # fit model
    model = ARMA(history, order=(4,0))
    model_fit = model.fit()
    # one step forecast
    yhat = model_fit.forecast()[0]
    # store forecast and ob
    predictions.append(yhat)
    history.append(test[t])
    print('predicted=%f, expected=%f' % (yhat, test[t]))

'''Evaluate forecasts'''
rmse = sqrt(mean_squared_error(test, predictions))

print('Test RMSE: %.3f' % rmse)

'''Plot forecast against actual outcomes'''
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


