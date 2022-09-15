import matplotlib.pyplot as plt 
plt.style.use('fast')
from pyFTS.partitioners import Grid, CMeans, Entropy
from pyFTS.common import Util as cUtil
import pyFTS.models.sadaei as sd
from pyFTS.models import hofts,pwfts
from pyFTS.benchmarks import Measures
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

def main(Dataset,df):
    train_size = int(len(Dataset.demand)*2/3)
    train_set = Dataset.demand[:train_size].to_list()
    test_set = Dataset.demand[train_size:]
    data = Dataset['demand'].values

    tdiff = Transformations.Differential(1)
    partitioner_grid = Grid.GridPartitioner(data=data,npart=15)
    partitioner_entropy = Entropy.EntropyPartitioner(data=data,npart=15)
    partitioner_cmean = CMeans.CMeansPartitioner(data=data,npart=15)
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=[15,5])
    partitioner_grid.plot(ax[0])
    partitioner_entropy.plot(ax[1])
    partitioner_cmean.plot(ax[2])
    plt.tight_layout()

    models=[]

    model = hofts.HighOrderFTS(partitioner=partitioner_grid,order=3)
    model.name='HOFTS'
    models.append(model)

    model = hofts.WeightedHighOrderFTS(partitioner=partitioner_grid,order=3)
    model.name='WHOFTS'
    models.append(model)

    df = createModels(models=models,train_set=train_set,test_set=test_set,df=df)
    
    return df
