from copy import Error
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
plt.style.use('fast')
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import seaborn as sns
from outliers import smirnov_grubbs as grubbs
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import ARIMA
import ExponentialSmoothing as ExpS
import FTS 

def grubbsTest(dataset,plot=True):
    array = dataset.demand.to_list()
    index = grubbs.max_test_indices(array,alpha=1) # other functions: min_test, max_test and max_test_outliers
    if plot:
        plt.figure()
        plt.plot(dataset)
        plt.plot(dataset.iloc[index],'g*')
        plt.xlabel('DateTime',fontsize=14)
        plt.ylabel('Power Demand (MW)',fontsize=14)
        plt.title('PJME Outliers')
    return dataset.iloc[index]


def init ():
    Dataset = pd.read_csv('./PJME_hourly.csv')
    Dataset.Datetime = pd.to_datetime(Dataset.Datetime)
    Dataset.set_index('Datetime', inplace = True)
    Dataset.rename(columns={'PJME_MW': 'demand'}, inplace=True)
    Dataset = Dataset.resample('H').mean()
    return Dataset

def EDA(Dataset):
    plt.figure()
    plt.plot(Dataset)
    plt.title('PJME Energy Consumption',fontsize=16)
    plt.xlabel('Datetime (h)',fontsize=14)
    plt.ylabel('Power Demand (MW)',fontsize=14)
    Dataset['weekday'] = Dataset.index.day_name()
    Dataset['hour'] = Dataset.index.hour

    # aggregated data
    _ = Dataset\
        .groupby(['hour', 'weekday'], as_index=False)\
        .agg({'demand':'median'})

    # plotting
    fig = px.line(_, 
                x='hour', 
                y='demand', 
                color='weekday', 
                title='Median Hourly Power Demand per Weekday',)
    fig.update_layout(xaxis_title='Hour',
                    yaxis_title='Energy Demand [MW]',
                    title_font_size=22,
                    font=dict(size=18))

    Dataset = Dataset.resample('D').mean()
    Dataset['weekday'] = Dataset.index.day_name()
    Dataset.boxplot(by='weekday', column=['demand'],grid=True)
    plt.xlabel('Weekday',fontsize=14)
    plt.ylabel('Power Demand (MW)',fontsize=14)
    plt.figure()
    sns.heatmap(Dataset.isnull(),cbar=False,yticklabels=False,cmap = 'viridis',)
    plt.title('Null/Nonexistent values on Dataset')
    plt.figure()
    sns.distplot(Dataset.demand,kde=True)
    plt.xlabel('Power Demand (MW)',fontsize=14)
    plt.ylabel("Density",fontsize=14)
    plt.title('Dataset Distribution',fontsize=16)
    Dataset.drop(['hour','weekday'],axis=1,inplace=True)
    fig.show()

    plot_acf(Dataset)
    plot_pacf(Dataset)

    result = seasonal_decompose(Dataset,model='additive')
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(result.observed)
    plt.ylabel('Observed')
    plt.title('PJME Seasonal Decomposition')
    plt.subplot(4,1,2)
    plt.plot(result.trend)
    plt.ylabel('Trend')
    plt.subplot(4,1,3)
    plt.plot(result.seasonal)
    plt.ylabel('Seasonality')
    plt.subplot(4,1,4)
    plt.plot(result.resid)
    plt.ylabel('Noise')

    train_size = int(len(Dataset.demand)*2/3)
    train_set = Dataset.demand[:train_size]
    test_set = Dataset.demand[train_size:]

    plt.figure()
    plt.plot(train_set)
    plt.plot(test_set)
    plt.title('Dataset')

    return Dataset,train_set,test_set

def Tests(Dataset):
    print(Dataset.describe())
    outliers = grubbsTest(Dataset,plot=False)

    result = adfuller(Dataset) # Série estacionária com p-value<<<<0.05 e ADF Statistic: -8.262177
    print('ADF Statistic: %.10f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    return outliers,result

def Orquestrate():
    dataset = init()
    dataset,train_set,test_set = EDA(dataset)
    Tests(dataset)
    Error_df = pd.DataFrame(columns = ["Name","RMSE","SMAPE","Theils U"])
    Error_df = ARIMA.main(train_set,test_set,Error_df)
    Error_df = ExpS.main(train_set,test_set,Error_df)
    Error_df = FTS.main(dataset,Error_df)
    print(Error_df)

if __name__=='__main__':
    os.chdir("C:/Users/CeSar/OneDrive/Documentos/TCC/DataBase/PJM Hourly Energy Consumption")
    size = (15,6)
    Orquestrate()
    plt.show()