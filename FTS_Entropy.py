import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('fast')
import warnings
warnings.filterwarnings('ignore')
from pyFTS.common import Transformations

os.chdir("C:/Users/CeSar/OneDrive/Documentos/TCC/DataBase/PJM Hourly Energy Consumption")
size = (15,6)

# -------------------------- Import do Dataset------------------------------------
Dataset = pd.read_csv('./PJME_hourly.csv')
Dataset.Datetime = pd.to_datetime(Dataset.Datetime)
Dataset.set_index('Datetime', inplace = True)
Dataset.rename(columns={'PJME_MW': 'demand'}, inplace=True)
Dataset = Dataset.resample('H').mean() # Resample para conerter dataframe inteiramente para hour steps

# -------------------------- Plot do Dataset------------------------------------
# plt.figure()
# plt.plot(Dataset)
# plt.title('PJME Energy Consumption')
# plt.xlabel('Datetime (h)')
# plt.ylabel('Power Demand (MW)')

# -------------------------- Plot do Dataset------------------------------------
Dataset['weekday'] = Dataset.index.day_name()
Dataset['hour'] = Dataset.index.hour
Dataset = Dataset.resample('D').mean() # Resample para corrigir valores faltantes
Dataset['weekday'] = Dataset.index.day_name()
# Dataset.boxplot(by='weekday', column=['demand'],grid=True)

# -------------------------- Processo de Introdução PyFTS------------------------------------
data = Dataset['demand'].values
tdiff = Transformations.Differential(1)
Xmin = Dataset['demand'].min()
Xmax = Dataset['demand'].max()
Universe = [Xmin,Xmax]
print(Universe) # análise preliminar do universo de discurso, será integrado uma margem de flutuação posteriormente

# ---------------------Análise de Modelo Fuzzy Time Series------------------------------------
from pyFTS.partitioners import Grid, CMeans, FCM, Entropy
from pyFTS.common import Util as cUtil
import pyFTS.models.sadaei as sd
from pyFTS.models import hofts,pwfts
from pyFTS.benchmarks import Measures

train_size = int(len(Dataset.demand)*2/3)
train_set = Dataset.demand[:train_size].to_list()
test_set = Dataset.demand[train_size:]

metodos = [Grid.GridPartitioner, Entropy.EntropyPartitioner, FCM.FCMPartitioner, CMeans.CMeansPartitioner ]

k = 15
rows = []

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[15,5])
ax.plot(train_set, label='Original',color='black')

for contador, metodo in enumerate(metodos):
  part = metodo(data=train_set, npart=k)
  model = hofts.HighOrderFTS(order=2, partitioner=part)
  model.fit(train_set)
  forecasts = model.predict(train_set)
  for o in range(model.order):
    forecasts.insert(0,None)
    
  ax.plot(forecasts[:-1], label=part.name)
  
  rmse, mape, u = Measures.get_point_statistics(train_set, model)
  rows.append([part.name, rmse, mape, u])

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))

pd.DataFrame(rows, columns=['Partitions','RMSE','MAPE','U'])

# partitioner = Grid.GridPartitioner(data=data,npart=20)
# partitioner_diff = Grid.GridPartitioner(data=data, npart=20, transformation=tdiff)
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15,5])
# partitioner.plot(ax)
# partitioner_diff.plot(ax) #Pesquisar a utilização de médias moveis 

# model1 = hofts.HighOrderFTS(partitioner=partitioner,order=2)
# model1.name='PJME'
# model1.fit(train_set, save_model=True, file_path='model1 PJME')
# result1 = Measures.get_point_statistics(test_set, model1)

# model2 = hofts.HighOrderFTS(partitioner=partitioner_diff,order=2)
# model2.name='PJME'
# model2.fit(train_set, save_model=True, file_path='model2 PJME')
# result2 = Measures.get_point_statistics(test_set, model2)

# model3 = hofts.WeightedHighOrderFTS(partitioner=partitioner,order=2)
# model3.name='PJME'
# model3.fit(train_set, save_model=True, file_path='model3 PJME')
# result3 = Measures.get_point_statistics(test_set, model3)

# model4 = hofts.WeightedHighOrderFTS(partitioner=partitioner_diff,order=2)
# model4.name='PJME'
# model4.fit(train_set, save_model=True, file_path='model4 PJME')
# result4 = Measures.get_point_statistics(test_set, model4)


# plt.figure()
# plt.plot(train_set)
# model1 = cUtil.load_obj('model1 PJME')
# forecasts = model1.predict(train_set)
# plt.plot(forecasts)
# plt.title('PJME')
# plt.tight_layout()

# plt.figure()
# plt.plot(train_set)
# model2 = cUtil.load_obj('model2 PJME')
# forecasts = model2.predict(train_set)
# plt.plot(forecasts)
# plt.title('PJME')
# plt.tight_layout()

# plt.figure()
# plt.plot(train_set)
# model3 = cUtil.load_obj('model3 PJME')
# forecasts = model3.predict(train_set)
# plt.plot(forecasts)
# plt.title('PJME')
# plt.tight_layout()

# plt.figure()
# plt.plot(train_set)
# model3 = cUtil.load_obj('model4 PJME')
# forecasts = model3.predict(train_set)
# plt.plot(forecasts)
# plt.title('PJME')
# plt.tight_layout()

# print(result1)
# print(result2)
# print(result3)
# print(result4)

plt.show()