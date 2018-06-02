import pandas as pd
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas.plotting import autocorrelation_plot
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def parser(x):
	return datetime.strptime(x, '%d/%m/%y %H')


data = pd.read_csv("BATADAL_dataset03.csv", header=0, parse_dates=[0], index_col=None, squeeze=True, date_parser=parser)
# data = pd.read_csv("BATADAL_dataset03.csv")
summary = data.groupby(['DATETIME', 'F_PU1']).mean()
# print(summary)

print(data.columns.values)


# The 'MS' string groups the data in buckets by start of the month
data = data.set_index('DATETIME').resample('D').mean()


autocorrelation_plot(summary)
plt.show()

# The term bfill means that we use the value before filling in missing values
data = data.fillna(data.bfill())


# print(data)

# print(data.head)
# data = data[["L_T1", "L_T2", "L_T3", "L_T4"]]

# data.plot()
# plt.show()
# p =  q = range(0, 1)
# pq = list(itertools.product(p,  q))
#


main_df = pd.DataFrame()

for c in data.columns[[0, 1, 2, 3, 4, 5, 6]]:
    # print(data[c])
    order_selection = sm.tsa.arma_order_select_ic(data[c].values, max_ar = 4, max_ma = 2, ic = "aic")
    ticker = [c]

    df_aic_min = pd.DataFrame([order_selection["aic_min_order"]], index=
    ticker)

    main_df = main_df.append(df_aic_min)
    main_df.to_csv("aic_min_orders.csv")



# arma_column15 = sm.tsa.ARMA(data['L_T1'].values, (4,0)).fit()
# print ("column1Arma5 {}".format(arma_column15.params))
# #
# arma_column17 = sm.tsa.ARMA(data['L_T1'].values, (3,2)).fit()
# print ("column1Arma7 {}".format(arma_column17.params))
#
# arma_column19 = sm.tsa.ARMA(data['L_T1'].values, (3,0)).fit()
# print ("column1Arma9 {}".format(arma_column17.params))
#
# arma_column21 = sm.tsa.ARMA(data['L_T1'].values, (2,2)).fit()
# print ("column1Arma9 {}".format(arma_column21.params))

arma_column15 = sm.tsa.ARMA(data['L_T2'].values, (3,1)).fit()
print ("column1Arma5 {}".format(arma_column15.params))
#
arma_column17 = sm.tsa.ARMA(data['L_T2'].values, (3,2)).fit()
print ("column1Arma7 {}".format(arma_column17.params))

arma_column19 = sm.tsa.ARMA(data['L_T2'].values, (3,0)).fit()
print ("column1Arma9 {}".format(arma_column17.params))

arma_column21 = sm.tsa.ARMA(data['L_T2'].values, (2,2)).fit()
print ("column1Arma9 {}".format(arma_column21.params))

print (arma_column15.aic, arma_column15.bic, arma_column15.hqic)
print (arma_column17.aic, arma_column17.bic, arma_column17.hqic)
print (arma_column19.aic, arma_column19.bic, arma_column19.hqic)
print (arma_column21.aic, arma_column21.bic, arma_column21.hqic)

# print(sm.stats.durbin_watson(arma_column15.resid.values))

X = data['L_T2'].values
size = int(len(X) * 0.66)
print(size)
train, test = X[0: size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation

for t in range(len(test)):
    # fit model
    model = ARIMA(history, order=(4,0,0))
    model_fit = model.fit()
    # one step forecast
    yhat = model_fit.forecast()[0]
    # store forecast and ob
    predictions.append(yhat)
    history.append(test[t])
    print('predicted=%f, expected=%f' % (yhat, test[t]))

# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))

print('Test RMSE: %.3f' % rmse)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


