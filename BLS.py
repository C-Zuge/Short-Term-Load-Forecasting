import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
plt.style.use('fast')
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm

os.chdir("C:/Users/CeSar/OneDrive/Documentos/TCC/DataBase/PJM Hourly Energy Consumption")
Dataset = pd.read_csv('./PJME_hourly.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
print(Dataset)
Train_Data = Dataset[:135780]
Test_Data = Dataset[135781:]
X = np.linspace(0,len(Train_Data),len(Train_Data))
print(X)
mod_wls = sm.WLS(X,Train_Data, weights=1)
res_wls = mod_wls.fit()
prstd, iv_l, iv_u = wls_prediction_std(res_wls)
print(res_wls.summary())

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(X, res_wls.fittedvalues, 'g--.')
ax.plot(X, iv_u, 'g--', label="WLS")
ax.plot(X, iv_l, 'g--')
ax.legend(loc="best")

plt.show()