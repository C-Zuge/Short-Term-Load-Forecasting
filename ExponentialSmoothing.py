from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import matplotlib.pyplot as plt 
plt.style.use('fast')
import numpy as np

def mape(y_true, y_pred):
    """ Mean Absolute Percentage Error """
    y_true, y_pred = np.array(y_true), np.array(y_pred) # convert to numpy arrays
    pe = (y_true - y_pred) / y_true # take the percentage error
    ape = np.abs(pe) # take the absolute values
    mape = np.mean(ape) # quantify the performance in a single number

    return f'{mape*100:.2f}%'

def rmse(y_true, y_pred):
    """ Root Mean Square Error """
    y_true, y_pred = np.array(y_true), np.array(y_pred) # convert to numpy arrays
    mse = np.mean((y_true - y_pred)**2) # Mean Square Error
    rmse = np.sqrt(mse) # take root of mse

    return f'{rmse:.4f}'

def UStatistic(targets, forecasts):
    """
    Theil's U Statistic

    :param targets: 
    :param forecasts: 
    :return: 
    """
    l = len(targets)
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(forecasts, list):
        forecasts = np.array(forecasts)

    naive = []
    y = []
    for k in np.arange(0, l - 1):
        y.append(np.subtract(forecasts[k], targets[k]) ** 2)
        naive.append(np.subtract(targets[k + 1], targets[k]) ** 2)
    return np.sqrt(np.divide(np.sum(y), np.sum(naive)))

def main(train_set,test_set,df):
    ExpModel = ExponentialSmoothing(train_set,trend='additive',seasonal='multiplicative',seasonal_periods=7).fit()

    plt.figure()
    ExpModel.fittedvalues.plot()
    plt.plot(test_set)
    ExpModel.forecast(len(test_set)).plot()
    plt.title('Exponential Smoothing Model')
    plt.legend(['Train Set','Test Set','Holt-Winters'])

    y_hat = ExpModel.forecast(len(test_set))
    MAPE = mape(test_set,y_hat)
    RMSE = rmse(test_set,y_hat)
    U = UStatistic(test_set,y_hat)
    dictResult= {   'Name': 'Holt-Winters',
                    'RMSE': RMSE,
                    'SMAPE': MAPE,
                    'Theils U': U,
                    }
    df = df.append(dictResult,ignore_index=True)

    return df