import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
plt.style.use('fast')
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

def is_White_Noise(dataset):
    try:
        dataset = dataset/dataset.describe()['max']
        Mean = dataset.describe()['mean']
        Std = dataset.describe()['std']
        AC = acf(dataset,fft=False)
    except:
        print('Not a pandas Series Type!')
        return True
    if(Mean>0.1): # Does the Series have a non-zero mean?
        pass
    else:
        if(Std<1): # Does the variance change over time?
            pass
        else:
            for i in range(len(AC)-1): # Do values correlate with lag values?
                if AC[i] < 0.1:
                    return True
            return False
    return False

def single_exp_smoothing (time_serie, alpha):
    #yf = a*yt + (1-a)y_lag
    data_forecasted=[]
    for i in range(1,len(time_serie)):
        yf = alpha*time_serie[i] + (1-alpha)*time_serie[i-1]
        data_forecasted.append(yf)
    return data_forecasted

def double_exp_smoothing(time_serie, alpha, beta):
    yhat = [time_serie[0]] # first value is same as series
    for t in range(1, len(time_serie)):
        if t==1:
            F, T = time_serie[0], time_serie[1] - time_serie[0]
        F_n_1, F = F, alpha*time_serie[t] + (1-alpha)*(F+T)
        T = beta*(F-F_n_1)+(1-beta)*T
        yhat.append(F+T)
    return yhat

os.chdir("C:/Users/CeSar/OneDrive/Documentos/TCC/DataBase/PJM Hourly Energy Consumption")

Dataset = pd.read_csv('./PJME_hourly.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
Dataset = Dataset[:].rolling(7).mean()

print(is_White_Noise(Dataset))

plt.figure(figsize=(7,3))
plt.plot(Dataset)

'''
Mes 7-9 Muito Alta (Picos maiores até 60kW)
Mes 12-4 Alta (Picos menores até 48kW)
Outros Meses baixa (Dados entre 20kW e 35.5kW)
'''

Train_Data = Dataset[:'2004-12-31']
Test_Data = Dataset['2005-1-1':'2006-1-1']
for x in Train_Data:
    Train_Values = x
for x in Test_Data:
    Test_Values = x
#temp = pd.DataFrame(Dataset.values)
#dataframe = pd.concat([temp.shift(1), temp],axis=1)
#dataframe.columns = ['t', 't-1']

result = seasonal_decompose(Test_Data,model='additive',freq=365).plot()
#result = seasonal_decompose(Test_Data,model='multiplicative',freq=365).plot()
plt.figure(figsize=(7,3))
pd.plotting.autocorrelation_plot(Test_Data)
plt.figure(figsize=(7,3))
Dataset.hist()

plt.figure(figsize=(7,3))
t1 = single_exp_smoothing(Train_Data, 0.8) # Montar Dataframe incluindo tempo para depois fazer o plot
plt.plot(t1,'m',Test_Values,'r')

plt.figure(figsize=(7,3))
t2 = double_exp_smoothing(Train_Data, 0.5, 0.3) # Montar Dataframe incluindo tempo para depois fazer o plot
plt.plot(t2,'-m')

plt.show()
