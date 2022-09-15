import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('fast')
import warnings
warnings.filterwarnings('ignore')
from pyFTS.common import Transformations

def handleModel(train_set,model):
    plt.figure()
    plt.plot(train_set)
    modelX = cUtil.load_obj(model.name)
    forecasts = modelX.predict(train_set)
    plt.plot(forecasts)
    plt.title(model.name)
    plt.tight_layout()

def createModels(models,train_set,test_set,df):
    for model in models:
        model.fit(train_set, save_model=True, file_path=model.name)
        result = Measures.get_point_statistics(test_set, model)
        dictResult= {   'Name': model.name,
                        'RMSE': result[0],
                        'SMAPE': result[1],
                        'Theils U': result[2]
                    }
        df = df.append(dictResult,ignore_index=True)
        handleModel(model=model, train_set=train_set)
    return df
        
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
Dataset.boxplot(by='weekday', column=['demand'],grid=True)

# -------------------------- Processo de Introdução PyFTS------------------------------------
data = Dataset['demand'].values
tdiff = Transformations.Differential(1)
Xmin = Dataset['demand'].min()
Xmax = Dataset['demand'].max()
Universe = [(Xmin*0.8),(Xmax*1.2)]
print(Universe) # análise preliminar do universo de discurso, será integrado uma margem de flutuação posteriormente

# ---------------------Análise de Modelo Fuzzy Time Series------------------------------------
from pyFTS.partitioners import Grid, CMeans, Entropy
from pyFTS.common import Util as cUtil
import pyFTS.models.sadaei as sd
from pyFTS.models import hofts,pwfts
from pyFTS.benchmarks import Measures

train_size = int(len(Dataset.demand)*2/3)
train_set = Dataset.demand[:train_size].to_list()
test_set = Dataset.demand[train_size:]
models=[]
result = pd.DataFrame(columns = ["Name","RMSE","SMAPE","Theils U"])

partitioner_grid = Grid.GridPartitioner(data=data,npart=15)
partitioner_entropy = Entropy.EntropyPartitioner(data=data,npart=15)
partitioner_cmean = CMeans.CMeansPartitioner(data=data,npart=15)
partitioner_diff = Grid.GridPartitioner(data=data, npart=15, transformation=tdiff)
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=[15,5])
partitioner_grid.plot(ax[0])
partitioner_entropy.plot(ax[1])
partitioner_cmean.plot(ax[2])
plt.tight_layout()

model = hofts.HighOrderFTS(partitioner=partitioner_grid,order=3)
model.name='Model1 PJME'
models.append(model)

model = hofts.HighOrderFTS(partitioner=partitioner_diff,order=3)
model.name='Model2 PJME'
models.append(model)

model = hofts.WeightedHighOrderFTS(partitioner=partitioner_grid,order=3)
model.name='Model3 PJME'
models.append(model)

model = hofts.WeightedHighOrderFTS(partitioner=partitioner_diff,order=3)
model.name='Model4 PJME'
models.append(model)

result = createModels(models=models,train_set=train_set,df=result)
print(result)

plt.show()