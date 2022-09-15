import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pyFTS.data import Enrollments
from pyFTS.partitioners import Grid
from pyFTS.models import chen
import seaborn as sns

#                       Dataset handler process
# --------------------------------------------------------------------
os.chdir("C:/Users/CeSar/OneDrive/Documentos/TCC/DataBase/PJM Hourly Energy Consumption")
Dataset = pd.read_csv('./PJME_hourly.csv')
Dataset.Datetime = pd.to_datetime(Dataset.Datetime)
Dataset.set_index('Datetime', inplace = True)
Dataset.rename(columns={'PJME_MW': 'demand'}, inplace=True)
Dataset = Dataset.resample('H').mean()
plt.figure()
plt.plot(Dataset)
plt.title('PJME Energy Consumption')
plt.xlabel('Datetime (h)')
plt.ylabel('Power Demand (MW)')
Dataset = Dataset.resample('D').mean()
Dataset['weekday'] = Dataset.index.day_name()

# --------------------------------------------------------------------

# train_size = int(len(Dataset.demand)*2/3)
# train_set = Dataset.demand[:train_size].to_list()
# test_set = Dataset.demand[train_size:].to_list()

# #Universe of Discourse Partitioner
# partitioner = Grid.GridPartitioner(data=train_set,npart=7)
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15,5])
# partitioner.plot(ax)

# # Create an empty model using the Chen(1996) method
# model = chen.ConventionalFTS(partitioner=partitioner)

# # The training procedure is performed by the method fit
# model.fit(train_set)

# #Print the model rules
# print(model)

# # The forecasting procedure is performed by the method predict
# forecasts = model.predict(test_set)

# #Plot 
# plt.figure()
# plt.plot(test_set)
# plt.plot(forecasts)
# plt.show()

# --------------------------------------------------------------------

from pyFTS.common import Transformations

tdiff = Transformations.Differential(1)
boxcox = Transformations.BoxCox(0)

from pyFTS.partitioners import Grid, Util as pUtil
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.models import chen

tag = 'chen_partitioning'
_type = 'point'


dataset = Dataset.demand.to_list()

bchmk.sliding_window_benchmarks(dataset, 1000, train=0.8, inc=0.2,
                                methods=[chen.ConventionalFTS],
                                benchmark_models=False,
                                transformations=[None],
                                partitions=np.arange(10,100,2), 
                                progress=False, type=_type,
                                distributed=False,
                                file="benchmarks.db", dataset='PJME', tag=tag)

bchmk.sliding_window_benchmarks(dataset, 1000, train=0.8, inc=0.2,
                                methods=[chen.ConventionalFTS],
                                benchmark_models=False,
                                transformations=[tdiff],
                                partitions=np.arange(3,30,1), 
                                progress=False, type=_type,
                                distributed=False,
                                file="benchmarks.db", dataset='PJME', tag=tag)

from pyFTS.benchmarks import Util as bUtil

df1 = bUtil.get_dataframe_from_bd("benchmarks.db",
                                  "tag = 'chen_partitioning' and measure = 'rmse'and transformation is null")

df2 = bUtil.get_dataframe_from_bd("benchmarks.db",
                                  "tag = 'chen_partitioning' and measure = 'rmse' and transformation is not null")

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[15,7])

g1 = sns.boxplot(x='Partitions', y='Value', hue='Dataset', data=df1, showfliers=False, ax=ax[0], 
                 palette="Set3")
box = g1.get_position()
g1.set_position([box.x0, box.y0, box.width * 0.85, box.height]) 
g1.legend(loc='right', bbox_to_anchor=(1.15, 0.5), ncol=1)
ax[0].set_title("Original data")
ax[0].set_ylabel("RMSE")
ax[0].set_xlabel("")

g2 = sns.boxplot(x='Partitions', y='Value', hue='Dataset', data=df2, showfliers=False, ax=ax[1], 
                 palette="Set3")
box = g2.get_position()
g2.set_position([box.x0, box.y0, box.width * 0.85, box.height]) 
g2.legend(loc='right', bbox_to_anchor=(1.15, 0.5), ncol=1)
ax[1].set_title("Differentiated data")
ax[1].set_ylabel("RMSE")
ax[1].set_xlabel("Number of partitions of the UoD")