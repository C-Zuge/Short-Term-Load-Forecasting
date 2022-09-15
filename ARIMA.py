from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
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

def ar(train_set,test_set,df):
    ar_model = AR(train_set,freq='D').fit(2)
    forecast = ar_model.params[0] + ar_model.params[1]*test_set.shift(1) + ar_model.params[2]*test_set.shift(2)
    RMSE = rmse(test_set['2013-01-24':],forecast['2013-01-24':])
    MAPE = mape(test_set['2013-01-24':],forecast['2013-01-24':])
    dictResult= {   'Name': 'AR 2',
                    'RMSE': RMSE,
                    'SMAPE': MAPE,
                    }
    df = df.append(dictResult,ignore_index=True)
    plt.plot(forecast)

    ar_model = AR(train_set,freq='D').fit(5)
    forecast = ar_model.params[0]
    for i, coef in enumerate(ar_model.params[1:]):
        forecast += coef * test_set.shift(i+1)
    RMSE = rmse(test_set['2013-01-27':],forecast['2013-01-27':])
    MAPE = mape(test_set['2013-01-27':],forecast['2013-01-27':])
    dictResult= {   'Name': 'AR 5',
                    'RMSE': RMSE,
                    'SMAPE': MAPE,
                    }
    df = df.append(dictResult,ignore_index=True)
    plt.plot(forecast)
    plt.title('AR Model')
    plt.legend(['Train Set','Test Set','AR2 Model','AR5 Model'])
    
    return df

def arma(train_set,test_set,df):
    
    config_arma = (5,4) # P=2 e Q=2 -> (p,q)
    arma_train = ARMA(train_set, freq='D', order=config_arma).fit()
    arma_test = ARMA(test_set, freq='D', order=config_arma).fit(arma_train.params)
    RMSE = rmse(test_set,arma_test.predict())
    MAPE = mape(test_set,arma_test.predict())
    dictResult= {   'Name': 'ARMA {},{}'.format(config_arma[0],config_arma[1]),
                    'RMSE': RMSE,
                    'SMAPE': MAPE,
                    }
    df = df.append(dictResult,ignore_index=True)
    plt.figure()
    plt.plot(train_set)
    plt.plot(test_set)
    plt.plot(arma_test.predict())
    plt.title('ARMA Model')
    plt.legend(['Train Set','Test Set','ARMA Model'])

    return df

def arima(train_set,test_set,df):
    from statsmodels.tsa.arima_model import ARIMA
    config_arima = (14,1,4) # P=14, D=1 e Q=4 -> (p,d,q)
    arima_train = ARIMA(train_set, freq='D',order=config_arima).fit()
    arima_test = ARIMA(test_set, freq='D',order=config_arima).fit(arima_train.params)
    y_hat = arima_train.forecast(len(test_set))
    plt.figure()
    plt.plot(train_set)
    plt.plot(test_set)
    plt.plot(arima_test.predict(typ='levels'))
    plt.title('ARIMA Model')
    plt.legend(['Train Set','Test Set','ARIMA Model'])

    print(y_hat[0])
    MAPE = mape(test_set,y_hat[0])
    RMSE = rmse(test_set,y_hat[0])
    dictResult= {   'Name': 'ARMA {},{},{}'.format(config_arima[0],config_arima[1],config_arima[2]),
                    'RMSE': RMSE,
                    'SMAPE': MAPE,
                    }
    df = df.append(dictResult,ignore_index=True)
    # plt.plot(arima.forecast(len(test_set))[0])

    return df

def main(train_set,test_set,df):
    df = ar(train_set,test_set,df)
    df = arma(train_set,test_set,df)
    df = arima(train_set,test_set,df)

    return df
