import numpy as np
import pandas as pd
import math

## 加载数据 并把['bodyType', 'model']转为int 
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
    data['salesVolume'], data['forecastVolum']
    data['bodyType'] = data['model'].map(train_sales_data.drop_duplicates('model').set_index('model')['bodyType'])

    for i in ['bodyType', 'model']:
        data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))

    data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']
    return data

## 生成平移特征
def genShitFeat(data,shift_list):
    shift_feat = []
    data['model_adcode'] = data['adcode'] + data['model']
    data['model_adcode_mt'] = data['model_adcode'] * 100 + data['mt']
    for i in shift_list:  ## 平移12个月
        shift_feat.append('shift_model_adcode_mt_label_{0}'.format(i))
        data['model_adcode_mt_{0}'.format(i)] = data['model_adcode_mt'] + i
        data_last = data[~data.label.isnull()].set_index('model_adcode_mt_{0}'.format(i))
        data['shift_model_adcode_mt_label_{0}'.format(i)] = data['model_adcode_mt'].map(data_last['label'])
    return data,shift_feat

def get_shift_feat2(df_,col_list,mt_list):   
    df = df_.copy()
    shift_feat = []
    df['model_adcode'] = df['adcode'] + df['model']
    df['model_adcode_mt'] = df['model_adcode'] * 100 + df['mt']
#     for col in tqdm(['label','popularity']):
    for col in col_list:
        # shift
        for i in mt_list:
            shift_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])    
    return df,shift_feat

## 生成统计特征
def genStatFeat(data,fea_list,target):
    df = data.copy()
    print('统计列的sum,mean,max,min,分位数0.2,分位数0.5,分位数0.8')
    stat_feat = []
    for f in fea_list:
        print('构造特征:',f)
        for t in target:
            print('构造特征:',f,'_',t)
            g1 = df.groupby([f])
            df1 = g1.agg({t:["sum","mean","max","min"]})
            df1.columns = ['{}_{}_sum'.format(t,f),'{}_{}_mean'.format(t,f),'{}_{}_max'.format(t,f),'{}_{}_mim'.format(t,f)]
            df1['{}_{}_median2'.format(t, f)] = g1['label'].quantile(0.2)
            df1['{}_{}_median5'.format(t, f)] = g1['label'].quantile(0.5)
            df1['{}_{}_median8'.format(t, f)] = g1['label'].quantile(0.8)
            df1.reset_index(inplace=True)
            df = df.merge(df1,'left',on=[f])
            stat_feat = stat_feat+list(df1.columns)
    return df,stat_feat

## 把预测log_label的cv概率转为提交文档格式
def genLogSub(data,res,n_split):
    res2 = pd.DataFrame()
    for i in range(1,n_split+1):  
        res2['prob_%s' % str(i)] = res['prob_%s' % str(i)].apply(lambda x : math.exp(x))
    sum_pred = res2.sum(axis=1) / 5
    sub = data[data['mt']>24][['id']]
    sub.reset_index(drop=True,inplace=True)
    sub['forecastVolum'] = sum_pred.astype(int)
    return sub
def featImport(lgb_model,features):
    df_imp = pd.DataFrame()
    imp = lgb_model.feature_importances_
    df_imp['features'] = features
    df_imp['import'] = imp
    df_imp.sort_values(by=['import'],ascending=False,inplace=True)
    return df_imp