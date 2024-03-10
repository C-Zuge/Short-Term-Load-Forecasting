import matplotlib.pyplot as plt 
plt.style.use('fast')
from pyFTS.partitioners import Grid, CMeans, Entropy
from pyFTS.common import Util as cUtil
from pyFTS.models import hofts,pwfts
from pyFTS.models.multivariate import mvfts, wmvfts, granular, variable
from pyFTS.benchmarks import Measures
from pyFTS.common import Transformations
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime
from pyFTS.common import Membership

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

def createModelsMV(models,train_set,test_set,df):
    pass

def main(Dataset,df):
    train_size = int(len(Dataset.demand)*2/3)
    train_set_pd = Dataset.demand[:train_size].to_list()
    train_set = Dataset.demand[:train_size].to_list()
    test_set = Dataset.demand[train_size:]
    data = Dataset['demand'].values

    # sp = {'seasonality': DateTime.day_of_year , 
    #   'names': ['Jan','Fev','Mar','Abr','Mai','Jun','Jul', 'Ago','Set','Out','Nov','Dez']}

    # vmonth = variable.Variable("Month", data_label="data", 
    #                         partitioner=seasonal.TimeGridPartitioner, npart=12, 
    #                         data=train_set_pd, partitioner_specific=sp)
    
    # sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k)+'hs' for k in range(0,24)]}

    # vhour = variable.Variable("Hour", data_label="data", 
    #                         partitioner=seasonal.TimeGridPartitioner, npart=24, 
    #                         data=train_set_pd, partitioner_specific=sp)
    
    # vload = variable.Variable("Demand", data_label="demand",
    #                      partitioner=Grid.GridPartitioner, npart=20, 
    #                      func=Membership.gaussmf, 
    #                      data=train_set_pd) 

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

    # model = mvfts.MVFTS(explanatory_variables=[vmonth, vhour, vload], target_variable=vload)
    # model.name='MVFTS'
    # models.append(model)

    # model = wmvfts.WeightedMVFTS(explanatory_variables=[vmonth, vhour, vload], target_variable=vload)
    # model.name='WMVFTS'
    # models.append(model)

    # model = granular.GranularWMVFTS(explanatory_variables=[vmonth, vhour, vload], target_variable=vload, 
    #                                 order=2, 
    #                                 knn=2, 
    #                                 fts_method=pwfts.ProbabilisticWeightedFTS, 
    #                                 fuzzyfy_mode='both')
    # model.name='GranularWMWFTS'
    # models.append(model)

    df = createModels(models=models,train_set=train_set,test_set=test_set,df=df)
    
    return df
