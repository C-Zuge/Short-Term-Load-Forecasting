import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
plt.style.use('fast')
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

os.chdir("C:/Users/CeSar/OneDrive/Documentos/TCC/DataBase/PJM Hourly Energy Consumption")
Dataset = pd.read_csv('./PJME_hourly.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
print(Dataset)
Train_Data = Dataset[:135780]
Test_Data = Dataset[135781:]

model = HWES(Train_Data, seasonal_periods=12, trend='add', seasonal='mul')
fitted = model.fit()
print(fitted.summary())
forecast = fitted.forecast(steps=12)
print(forecast)


fig = plt.figure()
past, = plt.plot(Train_Data.index, Train_Data, 'b.-')
future, = plt.plot(Test_Data.index, Test_Data, 'r.-')
predicted_future, = plt.plot(Test_Data.index, forecast, 'k.-', label='Forecast')
plt.show()