import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
plt.style.use('fast')
import seaborn as sns
import statsmodels.tsa.holtwinters as Hw

def movmean(dataset, n):
    return list(np.convolve(dataset,np.ones((n))/n,mode='full'))
def get_values(dataset):
    return [dataset[i][0] for i in range(len(dataset))]
def SES (time_serie, alpha, forecast):
    #yf = a*yt + (1-a)y_lag
    tam = len(time_serie)-1
    data_forecasted=[]
    y_curr = time_serie[tam-1]
    y_lag = time_serie[tam-2]
    data_forecasted.append(y_lag)
    data_forecasted.append(y_curr)
    for i in range(forecast):
        yf = alpha*y_curr + (1-alpha)*y_lag
        data_forecasted.append(yf)
        y_lag = y_curr
        y_curr=yf
    return data_forecasted
def DES(time_serie, alpha, forecast):
    return 0
        

os.chdir("C:/Users/CeSar/OneDrive/Documentos/TCC/DataBase/PJM Hourly Energy Consumption")

pjme = pd.read_csv('./PJME_hourly.csv',usecols=[1])

data_csv = pjme.dropna()
dataset = data_csv.values#.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
#dataset = get_values(list(map(lambda x: (x-min_value)/scalar, dataset)))
dataset = get_values(dataset)
Train=dataset[0:119085]
Test = dataset[119086:]
Train_month_data = movmean(Train,24)
Test_month_data = movmean(Test,24)
#plt.figure(figsize=(15,3))
#plt.plot(Test_month_data,'r')
#plt.figure(figsize=(15,3))
#plt.plot(Test_month_data,'r',Train_month_data,'b')

#------------------Simple Exponential Smoothing----------------------------
Fcast1 = SES(Train_month_data,0.6,len(Test_month_data))
plt.figure(figsize=(15,3))
plt.plot(Test_month_data,'m',Fcast1,'b')

Fcast2 = Hw.SimpleExpSmoothing(Train_month_data).fit().predict()
plt.figure(figsize=(15,3))
plt.plot(Test_month_data,'m',Fcast2,'b')

#------------------Double Exponential Smoothing----------------------------
Holtt = Hw.ExponentialSmoothing(Train_month_data,trend='multiplicative', damped=False, seasonal=None).fit()
Holtf = Holtt.forecast(steps=365)
plt.figure(figsize=(15,3))
plt.plot(Train_month_data,'m',Holtf,'k')
plt.show()