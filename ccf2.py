import numpy as np
import pandas as pd

def loaddata(path):
     ## 加载元数据
    train_sales_data = pd.read_csv(path + '/Train/train_sales_data.csv')
    train_search_data = pd.read_csv(path + '/Train/train_search_data.csv')
    train_user_reply_data = pd.read_csv(path + '/Train/train_user_reply_data.csv')

    test = pd.read_csv(path + '/evaluation_public.csv')
    
    ## join train 和 test

    data = pd.concat([train_sales_data, test], ignore_index=True)
    data = data.merge(train_search_data, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
    data = data.merge(train_user_reply_data, 'left', on=['model', 'regYear', 'regMonth'])
    
    
        
    data['yearmonth'] = data['regYear']*100+data['regMonth']
    data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']
    data['label'] = data['salesVolume']
    data['id'] = data['id'].fillna(0).astype(int)
    del data['salesVolume'], data['forecastVolum']
    data['bodyType'] = data['model'].map(train_sales_data.drop_duplicates('model').set_index('model')['bodyType'])
    ## 类别属性转为int
    for i in ['bodyType', 'model']:
        data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
    data['model_adcode'] = data['adcode'] + data['model']
    data['model_adcode_mt'] = data['model_adcode'] * 100 + data['mt']
    return data

def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby('model').agg({
        pred: list,
        label: [list, 'mean'],

    }).reset_index()

    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    for raw in data_agg[['{0}_list'.format(pred), '{0}_list'.format(label), '{0}_mean'.format(label)]].values:
        nrmse_score.append(
            mse(raw[0], raw[1]) ** 0.5 / raw[2]
        )
    print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)
