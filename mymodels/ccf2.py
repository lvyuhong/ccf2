import numpy as np
import pandas as pd

def loaddata(path):
    train_sales_data = pd.read_csv(path + '/Train/train_sales_data.csv')
    train_search_data = pd.read_csv(path + '/Train/train_search_data.csv')
    train_user_reply_data = pd.read_csv(path + '/Train/train_user_reply_data.csv')

    test = pd.read_csv(path + '/evaluation_public.csv')

    data = pd.concat([train_sales_data, test], ignore_index=True)
    data = data.merge(train_search_data, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
    data = data.merge(train_user_reply_data, 'left', on=['model', 'regYear', 'regMonth'])

    data['label'] = data['salesVolume']
    data['id'] = data['id'].fillna(0).astype(int)
    del data['salesVolume'], data['forecastVolum']
    data['bodyType'] = data['model'].map(train_sales_data.drop_duplicates('model').set_index('model')['bodyType'])

    for i in ['bodyType', 'model']:
        data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))

    data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']
    return data