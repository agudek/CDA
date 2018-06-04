from datetime import timedelta
from pandas import datetime
import statistics as st
import pandas as pd
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


def plot_arma(column_name, dates, y, ypred, std, mean):
    day_before = dates[0] - timedelta(days=10)
    last_day = dates[-1] - timedelta(days=10)

    upperbound = mean + (3 * std)
    lowerbound = mean - (3 * std)

    plt.axhline(y=(upperbound), color='red')
    plt.axhline(y=(lowerbound), color='red')

    plt.xlim(day_before, last_day)

    plt.plot(dates, y)
    plt.plot(dates, ypred)
    plt.xlabel("Date")
    plt.ylabel(column_name)
    plt.legend(loc="lower right")
    plt.savefig("armaResults/Anomaly%s" % column_name)
    plt.close()


def plot_residual(column_name, dates, residuals):
    residuals = pd.DataFrame(residuals)
    residuals.plot()
    plt.savefig("armaResults/Residual%s" % column_name)
    plt.close()

    residuals.plot(kind='kde')
    plt.savefig("armaResults/Density%s" % column_name)
    plt.close()
    print(residuals.describe())


def parser(x):
    return datetime.strptime(x, '%d/%m/%y %H')


def predict(coef, history):
    yhat = 0.0
    for i in range(1, len(coef) + 1):
        yhat += coef[i - 1] * history[-i]
    return yhat


def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)


def aic(dataset, columns):
    aic_orders = {}

    for column_name in columns:
        column = dataset[column_name]
        order_selection = arma_order_select_ic(column.values, max_ar = 4, max_ma = 2, ic = "aic")
        aic_orders[column_name] = order_selection.aic_min_order
    return aic_orders

train_data1 = pd.read_csv("BATADAL_dataset03.csv", header=0, parse_dates=[0], index_col=None, squeeze=True, date_parser=parser)
train_data2 = pd.read_csv("BATADAL_dataset04.csv", header=0, parse_dates=[0], index_col=None, squeeze=True, date_parser=parser)

columns = train_data1.columns[[1, 2, 3, 4, 5, 6, 7]]
aic_orders = aic(train_data1, columns)
predictions = {}

for c in columns:
    test_column_name = c
    series = train_data1[c].values
    train_data = [x for x in series]
    test_data_values = train_data2[test_column_name].values
    history = [x for x in test_data_values[:5]]
    test_data_values = test_data_values[5:]

    standard_deviation = st.stdev(train_data)
    mean = st.mean(train_data)

    predictions[c] = list()

    model = ARMA(train_data, order = (aic_orders[c]))
    model_fit = model.fit(disp=False)

    for t in range(len(test_data_values)):
        ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
        resid = model_fit.resid

        diff = difference(history)
        yhat = history[-1] + predict(ar_coef, diff) + predict(ma_coef, resid)
        predictions[c].append(yhat)
        obs = test_data_values[t]
        history.append(obs)
    mse = mean_squared_error(test_data_values, predictions[c])
    rmse = np.sqrt(mse)
    print('Test RMSE: %.3f for column %s' % (rmse, c))

    dates = [x for x in train_data2['DATETIME']]
    dates = dates[5:]

    train_dates = [x for x in train_data1['DATETIME']]

    plot_arma(c, dates, predictions[c], test_data_values, standard_deviation, mean)
    plot_residual(c, train_dates, model_fit.resid)

